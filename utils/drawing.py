import cv2
import config


def setup_window():
    """Setup window based on WINDOW_MODE configuration."""
    mode = config.WINDOW_MODE
    name = config.WINDOW_NAME

    if mode == "normal":
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)

    elif mode == "resizable":
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

    elif mode == "maximized":
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    elif mode == "fullscreen":
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    else:
        raise ValueError(f"Invalid WINDOW_MODE: {mode}")
