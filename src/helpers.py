import winreg
import logging

def GetKeyFromRegistry(keyName: str) -> str:
    """
    Retrieves the API key from HKEY_CURRENT_USER\\Software\\microlyth.
    """
    # The path where we will store microlyth-related secrets
    registryPath = r"Software\\Microlyth"
    value = None
        
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, registryPath, 0, winreg.KEY_READ) as hkey:
            value, _ = winreg.QueryValueEx(hkey, keyName)
    except FileNotFoundError:
        logging.error(f"❌ Error: The registry path '{registryPath}' or key '{keyName}' does not exist.")
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred: {e}")

    return value