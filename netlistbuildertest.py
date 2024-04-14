from netlistbuilder import NetList

net = NetList("TestBoost")

matrix1 = [    \
            [2, 1, 0, 0, 0], \
            [0, 1, 2, 0, 0], \
            [3, 0, 1, 0, 2], \
            [0, 0, 1, 2, 0], \
            [2, 0, 0, 1, 0], \
            [2, 0, 0, 1, 0], \
            
            ]

matrix2 = ['V', 'L', 'M', 'D', 'C', 'R']

net.generate(matrix1, matrix2)
net.run()
net.plot()