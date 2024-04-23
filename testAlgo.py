from LongestPrefix import LongestPrefix



obj =  LongestPrefix()
with open("./outpu.txt",'r') as file:
    data = file.read().split("\n")
for d in data:
    obj.insert(d)