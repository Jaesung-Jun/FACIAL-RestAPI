from constants import Colors
import time

def logger(message, log_time = True, level="INFO"):
    now = time
    if level == "INFO":
        color = Colors.GREEN
        str = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {color}{message}{Colors.RESET}"
        print(str)
    elif level == "EXE_START":    #코드 실행중일때
        color = Colors.YELLOW
        str = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {color}{message}{Colors.RESET}"
        print(str)
    elif level == "EXE_FINISH":
        color = Colors.MAGENTA
        str = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {color}{message}{Colors.RESET}"
        print(str)
    elif level == "ERROR":
        color = Colors.RED
        str = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {color}{message}{Colors.RESET}"
        print(str)
    elif level == "DEBUG":
        color = Colors.BLUE
        str = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {color}{message}{Colors.RESET}"
        print(str)
    else:
        print("CHECK LOG SYNTAX")
        color = Colors.BLUE
        str = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {color}{message}{Colors.RESET}"
        print(str)