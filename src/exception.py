import sys
from src.logger import logging

# This function builds a detailed error message using exception info
def error_message_detailed(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in script: [{file_name}] at line [{exc_tb.tb_lineno}] with message: [{str(error)}]"
    return error_message

# Custom exception class that formats and logs the error clearly
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detailed(error_message, error_detail)

    def __str__(self):
        return self.error_message

if __name__ =="__main__":
    try:
        a = 1 / 0

    except Exception as e:
        logging.info("Divison by Zero")
        raise CustomException(e,sys)