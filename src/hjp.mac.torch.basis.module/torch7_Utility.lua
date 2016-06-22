-- Utility functions

-- torch.class
do 
  local Foo = torch.class("Foo")
  function Foo:__init()
    self.contents = 'this is some text'
  end
  
  function Foo:print()
    print(self.contents)
  end
  
  function Foo:bip()
    print('bip')
  end
end

foo = Foo()

foo:print()

do 
  local Bar, parent = torch.class('torch.Bar', 'Foo')
  function Bar:__init(stuff)
    parent.__init(self)
    self.stuff = stuff
  end
  
  function Bar:boing()
    print('boing!')
  end
  
  function Bar:print()
    print(self.contents)
    print(self.stuff)
  end
end

bar = torch.Bar('ha ha!')
bar:print()
bar:boing()
bar:bip()

-- torch.type()
print(torch.type(torch.Tensor()))
print(torch.type({}))
print(torch.type(7))

-- torch.typename()
print(torch.typename(torch.Tensor()))
print(torch.typename({}))
print(torch.typename(7))

-- torch.getmetatable()
for k, v in pairs(torch.getmetatable('torch.CharStorage')) do print(k, v) end

-- torch.totable()
print(torch.totable(torch.Tensor({1, 2, 3})))