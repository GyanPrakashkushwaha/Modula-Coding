"""The sys module in Python provides various functions and variables that are used to manipulate different 
parts of the Python runtime environment. It allows operating on the interpreter as it provides access to the variables and functions that interact
 strongly with the interpreter. Lets consider the below example."""

import sys
from src.logger import logging

def error_msg_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info() # this is execution info this gives 3 details and the 3rd variable exc_tb Gives below written 'error_msg'


    file_name = exc_tb.tb_frame.f_code.co_filename # this is for file name 
    
    error_msg = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(

    file_name, # this is for file number means [0]
    exc_tb.tb_lineno, # line number [1]
    str(error) # exact error 
    )

    return error_msg


class CustomException(Exception):
    def __init__(self, error_message,error_detail:sys) :
        super().__init__(error_message)
        self.error_message = error_msg_detail(error=error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message



# if __name__ == "__main__":

#     try:
#         a = 1/0
#     except Exception as e:
#         logging.info("ZERO Division error")
#         raise CustomException(e,sys)
    