import os

def separate_values(valu):
    value="".join(valu)
    rasd = value.split(sep='-')
    return rasd[0]




with open('data.txt', 'r') as f:
    lines = f.readlines()
    
last_line = lines[-1].strip()  # Убираем возможные пробелы и переносы строки


print(last_line)
x1=separate_values(last_line)
print(x1)
num_img = int(x1) + 1
f.close()
f=open('data.txt', 'a')
while True:
    # Пример ввода: "123-456"
    inp = input(f"{num_img}-")
    
    if inp != "q":
        if int(inp)>=0 and int(inp)<8: 
            values = f"{num_img}-" + separate_values(inp)
            f.write(values+'\n')
            num_img+=1
        else:
            print("неправильный формат ввода\n попробуй ещё раз")
    else:
        break