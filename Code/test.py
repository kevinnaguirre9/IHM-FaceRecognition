import numpy as np
import pandas as pd
from datetime import datetime

def markAttendance(path, user_id, name = "ELON MUSK"):
     df = pd.read_csv(path)
     now = datetime.now()
     current_date = now.strftime("%d/%m/%Y")
     current_time = now.strftime("%H:%M:%S") 
     filtered_df = df[(df['userID'] == 3)]
     print(filtered_df)

markAttendance("../TempData/Attendance.csv", 3)
