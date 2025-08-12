import os
from scilab2py import Scilab2Py

os.environ['SCI'] = r"C:\Users\stajyer1\AppData\Local\scilab-2025.1.0"
os.environ['PATH'] += os.pathsep + r"C:\Users\stajyer1\AppData\Local\scilab-2025.1.0\bin"

sci = Scilab2Py()
sci.eval("disp('Scilab bulundu ve bağlandı!')")
