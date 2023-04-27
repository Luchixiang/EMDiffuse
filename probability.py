import random

count = 0
month = [1,2,3,4,5,6,7,8,9,10,11,12]
true_count = 0
while count < 10000:
    random_list = []
    for _ in range(25):
        n = random.randint(1, 12)
        random_list.append(n)
    # print(random_list)
    flag = True
    for num in month:
        if num not in random_list:
            flag = False
    if flag:
        true_count += 1
print(true_count / count)