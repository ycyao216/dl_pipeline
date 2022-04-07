import os

ids = set()
for i in os.listdir("train/label"):
    name = os.path.join("train/label", i)
    files = open(name, "r")
    line_int = [int(j) for j in files.readlines()]
    for k in line_int:
        ids.add(k)

print(ids)
print(len(ids))
