-- File
require('torch')

file = torch.MomoryFile()
file:writeObject(object)
file:seek(1)
objectClone = file:readObject()

array = {}
x = torch.Tensor(1)
table.insert(array, x)
table.insert(array, x)

array[1][1] = 3.14
file = torch.DiskFile('foo.asc', 'w')
file:writeObject(array)
file:close()

file = torch.DiskFile('foo.asc', 'r')
arrayNew = file:readObject()

arrayNew[1][1] = 2.72
print(arrayNew)