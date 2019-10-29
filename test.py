def caffeinated(aList, bList):
	result = []	
	for a in aList:
		to_add = false
		for b in bList:
			if a[1] == b or a[2] == b:
				to_add = true
		if to_add == true:
			result.append(a[0])

aList = [("boba", "OVERrated", "sWEET"), ("COFFEE", "gross", "hOt"), ("tea", "DeCent", "alRight")]
bList = ["sweet", "GROSS", "hot", "overRATED"]

print(caffeinated(aList, bList))
#("boba", "COFFEE")

aList = [("monster", "bad", "gross"), ("RED bull", "ew", "why"),
("water", "good", "like ACTUALLY the best")]
bList = ["like actually the best", "GOOD"]
print(caffeinated(aList, bList))
#("water",)
aList = [("monster", "bad", "gross"), ("RED bull", "ew", "why"), ("water", "good", "like ACTUALLY the best")]
bList = ["good", "GROSS"]
print(caffeinated(aList, bList))
# ()
