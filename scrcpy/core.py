import os
import socket
import struct
import threading
import time
from enum import Enum
from time import sleep
from typing import Any, Callable, Optional, Tuple, Union

import cv2
import numpy as np
from adbutils import AdbConnection, AdbDevice, AdbError, Network, adb
from av.codec import CodecContext
from av.error import InvalidDataError

from .const import (
    EVENT_DISCONNECT,
    EVENT_FRAME,
    EVENT_INIT,
    LOCK_SCREEN_ORIENTATION_UNLOCKED,
)
from .control import ControlSender


class KeyboardInputModes(Enum):
    SDK = "sdk"
    UHID = "uhid"
    AOA = "aoa"
    DISABLED = "disabled"


class Client:

    def __init__(
        self,
        device: Optional[Union[AdbDevice, str, any]] = None,
        max_width: int = 0,
        video_bitrate: int = 8000000,
        audio_bitrate: Optional[int] = 128000,
        max_fps: int = 0,
        flip: bool = False,
        block_frame: bool = False,
        stay_awake: bool = False,
        lock_screen_orientation: int = LOCK_SCREEN_ORIENTATION_UNLOCKED,
        connection_timeout: int = 3000,
        video_encoder_name: Optional[str] = None,
        audio_encoder_name: Optional[str] = None,
        video_codec: Optional[str] = None,
        control_enabled: bool = True,
        show_touches: bool = False,
        power_off_on_close: bool = False,
        crop: Optional[Tuple[int, int, int, int]] = None,
        display_id: Optional[int] = None,
        display_buffer: Optional[int] = None,
        v4l2_buffer: Optional[int] = None,
        audio_buffer: Optional[int] = None,
        record: Optional[str] = None,
        no_playback: bool = False,
        no_video_playback: bool = False,
        keyboard: KeyboardInputModes = KeyboardInputModes.SDK,
        no_audio: bool = False,
    ):
        """
        Create a scrcpy client, this client won't be started until you call the start function

        Args:
            device: Android device, select first one if none, from serial if str
            max_width: frame width that will be broadcast from android server
            video_bitrate: The required bitrate for the video stream
            max_fps: maximum fps, 0 means not limited (supported after android 10)
            flip: flip the video
            block_frame: only return nonempty frames, may block cv2 render thread
            stay_awake: keep Android device awake
            lock_screen_orientation: lock screen orientation, LOCK_SCREEN_ORIENTATION_*
            connection_timeout: timeout for connection, unit is ms
            encoder_name: encoder name, enum: [OMX.google.h264.encoder, OMX.qcom.video.encoder.avc, c2.qti.avc.encoder, c2.android.avc.encoder], default is None (Auto)
        """
        # Check Params
        assert max_width >= 0, "max_width must be greater than or equal to 0"
        assert video_bitrate >= 0, "video bitrate must be greater than or equal to 0"
        assert audio_bitrate is None or audio_bitrate >= 0, "audio bitrate must be greater than or equal to 0"
        assert max_fps >= 0, "max_fps must be greater than or equal to 0"
        assert (
            -1 <= lock_screen_orientation <= 3
        ), "lock_screen_orientation must be LOCK_SCREEN_ORIENTATION_*"
        assert (
            connection_timeout >= 0
        ), "connection_timeout must be greater than or equal to 0"
        # assert encoder_name in [
        #     None,
        #     "OMX.google.h264.encoder",
        #     "OMX.qcom.video.encoder.avc",
        #     "c2.qti.avc.encoder",
        #     "c2.android.avc.encoder",
        # ]
        assert video_codec in [
            None,
            "h264",
            "h265",
            "av1"
        ]

        # Params
        self.flip = flip
        self.max_width = max_width
        self.video_bitrate = video_bitrate
        self.audio_bitrate = audio_bitrate
        self.max_fps = max_fps
        self.block_frame = block_frame
        self.stay_awake = stay_awake
        self.lock_screen_orientation = lock_screen_orientation
        self.connection_timeout = connection_timeout
        self.video_encoder_name = video_encoder_name
        self.audio_encoder_name = audio_encoder_name
        self.video_codec = video_codec
        self.control_enabled = control_enabled
        self.show_touches = show_touches
        self.power_off_on_close = power_off_on_close
        self.crop = crop
        self.display_id = display_id
        self.display_buffer = display_buffer
        self.v4l2_buffer = v4l2_buffer
        self.audio_buffer = audio_buffer
        self.record = record
        self.no_playback = no_playback
        self.no_video_playback = no_video_playback
        self.keyboard = keyboard
        self.no_audio = no_audio

        # Connect to device
        if device is None:
            device = adb.device_list()[0]
        elif isinstance(device, str):
            device = adb.device(serial=device)

        self.device = device
        self.listeners = dict(frame=[], init=[], disconnect=[])

        # User accessible
        self.last_frame: Optional[np.ndarray] = None
        self.resolution: Optional[Tuple[int, int]] = None
        self.device_name: Optional[str] = None
        if control_enabled:
            self.control = ControlSender(self)

        # Need to destroy
        self.alive = False
        self.__server_stream: Optional[AdbConnection] = None
        self.__video_socket: Optional[socket.socket] = None
        self.control_socket: Optional[socket.socket] = None
        if control_enabled:
            self.control_socket_lock = threading.Lock()

        # Available if start with threaded or daemon_threaded
        self.stream_loop_thread = None

    def __init_server_connection(self) -> None:
        """
        Connect to android server, there will be two sockets, video and control socket if it is enabled.
        This method will set: video_socket, control_socket, resolution variables
        """
        for _ in range(self.connection_timeout // 100):
            try:
                self.__video_socket = self.device.create_connection(
                    Network.LOCAL_ABSTRACT, "scrcpy"
                )
                break
            except AdbError:
                sleep(0.1)
                pass
        else:
            raise ConnectionError("Failed to connect scrcpy-server after 3 seconds")

        dummy_byte = self.__video_socket.recv(1)
        if not len(dummy_byte) or dummy_byte != b"\x00":
            raise ConnectionError("Did not receive Dummy Byte!")

        if self.control_enabled:
            self.control_socket = self.device.create_connection(
                Network.LOCAL_ABSTRACT, "scrcpy"
            )

        self.device_name = self.__video_socket.recv(64).decode("utf-8").rstrip("\x00")
        if not len(self.device_name):
            raise ConnectionError("Did not receive Device Name!")

        res = self.__video_socket.recv(4)
        self.resolution = struct.unpack(">HH", res)
        self.__video_socket.setblocking(False)

    def __deploy_server(self) -> None:
        """
        Deploy server to android device
        """
        jar_name = "scrcpy-server.jar"
        server_file_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), jar_name
        )
        self.device.sync.push(server_file_path, f"/data/local/tmp/{jar_name}")
        commands = [
            f"CLASSPATH=/data/local/tmp/{jar_name}",
            "app_process",
            "/",
            "com.genymobile.scrcpy.Server",
            "2.4",  # Scrcpy server version
            "log_level=info",
            f"max_size={self.max_width}",
            f"max_fps={self.max_fps}",
            f"video_bit_rate={self.bitrate}",
            "tunnel_forward=true",
            "send_frame_meta=false",
            f"control={self.control_enabled}",
            f"show_touches={self.show_touches}",
            f"stay_awake={self.stay_awake}",
            f"power_off_on_close={self.power_off_on_close}",
            "clipboard_autosync=false",
            f"crop={self.crop}",
            f"display_id={self.display_id}",
            f"keyboard={self.keyboard.value}"
        ]
        # Adding optionals
        if self.video_encoder_name:
            commands += [f"video_encoder={self.video_encoder_name}"]
        if self.audio_encoder_name:
            commands += [f"audio_encoder={self.audio_encoder_name}"]
        if self.video_codec:
            commands += [f"video_codec={self.video_codec}"]
        if self.display_id:
            commands += [f"display_id={self.display_id}"]
        if self.crop:
            commands += [f"crop={self.crop[0]}:{self.crop[1]}:{self.crop[2]}:{self.crop[3]}"]
        if self.display_buffer:
            commands += [f"display_buffer={self.display_buffer}"]
        if self.v4l2_buffer:
            commands += [f"v4l2_buffer={self.v4l2_buffer}"]
        if self.audio_buffer:
            commands += [f"audio_buffer={self.audio_buffer}"]
        if self.record:
            commands += [f"record={self.record}"]
        if self.no_playback:
            commands += ["no_playback"]
        if self.no_video_playback:
            commands += ["no_video_playback"]
        if self.no_audio:
            commands += ["no_audio"]
        if self.audio_bitrate:
            commands += [f"audio_bitrate={self.audio_bitrate}"]


        self.__server_stream: AdbConnection = self.device.shell(
            commands,
            stream=True,
        )

        # Wait for server to start
        self.__server_stream.read(10)

    def list_encoders(self):
        """
        Deploys the server and asks for the list of encoders
        Returns:

        """

    def start(self, threaded: bool = False, daemon_threaded: bool = False) -> None:
        """
        Start listening video stream

        Args:
            threaded: Run stream loop in a different thread to avoid blocking
            daemon_threaded: Run stream loop in a daemon thread to avoid blocking
        """
        assert self.alive is False

        self.__deploy_server()
        self.__init_server_connection()
        self.alive = True
        self.__send_to_listeners(EVENT_INIT)

        if threaded or daemon_threaded:
            self.stream_loop_thread = threading.Thread(
                target=self.__stream_loop, daemon=daemon_threaded
            )
            self.stream_loop_thread.start()
        else:
            self.__stream_loop()

    def stop(self) -> None:
        """
        Stop listening (both threaded and blocked)
        """
        self.alive = False
        if self.__server_stream is not None:
            try:
                self.__server_stream.close()
            except Exception:
                pass

        if self.control_socket is not None:
            try:
                self.control_socket.close()
            except Exception:
                pass

        if self.__video_socket is not None:
            try:
                self.__video_socket.close()
            except Exception:
                pass

    def __stream_loop(self) -> None:
        """
        Core loop for video parsing
        """
        codec = CodecContext.create(self.video_codec, "r")
        while self.alive:
            try:
                raw_stream = self.__video_socket.recv(0x10000)
                if raw_stream == b"":
                    raise ConnectionError("Video stream is disconnected")
                packets = codec.parse(raw_stream)
                for packet in packets:
                    frames = codec.decode(packet)
                    for frame in frames:
                        frame = frame.to_ndarray(format="bgr24")
                        if self.flip:
                            frame = cv2.flip(frame, 1)
                        self.last_frame = frame
                        self.resolution = (frame.shape[1], frame.shape[0])
                        self.__send_to_listeners(EVENT_FRAME, frame)
            except (BlockingIOError, InvalidDataError):
                time.sleep(0.01)
                if not self.block_frame:
                    self.__send_to_listeners(EVENT_FRAME, None)
            except (ConnectionError, OSError) as e:  # Socket Closed
                if self.alive:
                    self.__send_to_listeners(EVENT_DISCONNECT)
                    self.stop()
                    raise e

    def add_listener(self, cls: str, listener: Callable[..., Any]) -> None:
        """
        Add a video listener

        Args:
            cls: Listener category, support: init, frame
            listener: A function to receive frame np.ndarray
        """
        self.listeners[cls].append(listener)

    def remove_listener(self, cls: str, listener: Callable[..., Any]) -> None:
        """
        Remove a video listener

        Args:
            cls: Listener category, support: init, frame
            listener: A function to receive frame np.ndarray
        """
        self.listeners[cls].remove(listener)

    def __send_to_listeners(self, cls: str, *args, **kwargs) -> None:
        """
        Send event to listeners

        Args:
            cls: Listener type
            *args: Other arguments
            *kwargs: Other arguments
        """
        for fun in self.listeners[cls]:
            fun(*args, **kwargs)
