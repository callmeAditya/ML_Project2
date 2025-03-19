import sys ### it controls exception handling
import logging

def error_message_detail(er,er_detail:sys):
    _,_,exc_tb = er_detail.exc_info() ### it returns the exception type, value and traceback at which lines
    file_name = exc_tb.tb_frame.f_code.co_filename ### it returns the file name
    line_no = exc_tb.tb_lineno ### it returns the line number
    
    error_message = "Error occured in python script [{0}] line number [{1}] error message [{2}]".format(
        file_name,line_no,str(er)
    )
    return error_message


class MyException(Exception): ## inheret the Exception class
    def __init__(self, error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail) ### it returns the error message
        
    
    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        a = 10/0
    except Exception as e:
        logging.info("cannot divide by zero")
        raise MyException(e,sys)
        
        