## import modules here 

################# Question 0 #################

def add(a, b): # do not change the heading of the function
    return a + b


################# Question 1 #################

def nsqrt(x): # do not change the heading of the function
    if (x < 2):
        return x;
    else:
        small = nsqrt(x // 4) * 2
        large = small + 1
        
        if (large * large > x):
            return small
        else:
            return large


################# Question 2 #################


# x_0: initial guess
# EPSILON: stop when abs(x - x_new) < EPSILON
# MAX_ITER: maximum number of iterations

## NOTE: you must use the default values of the above parameters, do not change them

def find_root(f, fprime, x_0=1.0, EPSILON = 1E-7, MAX_ITER = 1000): # do not change the heading of the function
    
    x_new = x_0
    
    for index in range(MAX_ITER):
        x = x_new
        x_new = x - f(x) / fprime(x)
        
        if abs(x - x_new) < EPSILON:
            break
            
        index += 1
        
    return x_new



################# Question 3 #################

class Tree(object):
    def __init__(self, name='ROOT', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)


def make_sub_tree(parent_tree, tokens, start, end):
    if start <= end and parent_tree != None:
        i = 0
        while start + i <= end:
            sub_tree = None
            while start + i <= end and tokens[start + i] != '[' and tokens[start + i] != ']':       
                sub_tree = Tree(tokens[start + i])
                parent_tree.add_child(sub_tree)
                i += 1

            if start + i <= end:
               if tokens[start + i] == '[':
                   i += 1
                   i += make_sub_tree(sub_tree, tokens, start + i, end)
               elif tokens[start + i] == ']':
                   i += 1
                   return i
        return i
    return 0



def make_tree(tokens): # do not change the heading of the function
    if len(tokens) > 0:
        the_tree = Tree(tokens[0])
        if len(tokens) > 1 and tokens[1] == '[':
            make_sub_tree(the_tree, tokens, 1, len(tokens) - 1)
        return the_tree
    else:
        return None


def max_depth(root): # do not change the heading of the function
    depth = 0
    if None == root:
        return depth

    depth += 1
    max_sub_depth = 0;
    if len(root.children) > 0:
        for child in root.children:
            sub_depth = max_depth(child)
            if sub_depth > max_sub_depth:
                max_sub_depth = sub_depth

    return depth + max_sub_depth