import random
choose = ["기획", "개발", "디자인"]
personlist = ["은별", "예진", "이진", "채송"]

person = dict()

person["한나"] = choose[2]

# whofirst = random.randint(0,3)
random.shuffle(personlist)
random.shuffle(choose)
for x in personlist:
    randomnum = random.randint(0,2)
    person[x] = choose[randomnum]
    if (x == "은별" or "이진")and choose[randomnum] == "디자인":
        if randomnum > 0:
            person[x] = choose[randomnum -1]
        else:
            person[x] = choose[randomnum + 1]
    if (x == "예진") and choose[randomnum] == "기획":
        if randomnum > 0:
            person[x] = choose[randomnum -1]
        else:
            person[x] = choose[randomnum + 1]

print(person)






