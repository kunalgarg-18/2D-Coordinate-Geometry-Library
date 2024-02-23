import math
import sympy
import numpy as np
import matplotlib.pyplot as plt

class Point:

    def __init__(self, x, y):
        self._x = x
        self._y = y
    
    def __str__(self): 
        return f"({self._x},{self._y})"

    def getPoint(self):
        return self._x, self._y
    
    def setPoint(self, x, y):
        self._x = x
        self._y = y

    def midPoint(self, other):
        x1, y1 = self.getPoint()
        x2, y2 = other.getPoint()
        x = (x1+x2)/2
        y = (y1+y2)/2

        return Point(x,y)
    
    def sectionPoint(self, other, m, n):
        x1, y1 = self.getPoint()
        x2,y2  = other.getPoint()
        x = (n*x1 + m*x2)/(m+n)
        y = (n*y1 + m*y2)/(m+n)

        return Point(x, y)

    def integralPoints(self, other):
        x1,y1 = self.getPoint()
        x2,y2 = other.getPoint()

        if x1 == x2: return abs(y1 - y2) - 1
        elif y1 == y2: return abs(x1 - x2) - 1
        else: return math.gcd(abs(x1-x2), abs(y1-y2)) - 1 
    
    def getLinePointSlope(self, m):
        x1,y1 = self.getPoint()
        x,y = sympy.symbols('x y')
        e = sympy.Eq((y-y1)-m*x+m*x1, 0)
        return e

    def getLineTwoPoint(self, other):
        x1,y1 = self.getPoint()
        x2,y2 = other.getPoint()
        m = (y2-y1)/(x2-x1)
        x,y = sympy.symbols('x y')
        e = sympy.Eq((y-y1)-m*x+m*x1,0)
        return e

class Distance:

    def __init__(self, obj1:Point, obj2:Point):
        self._x1, self._y1 = obj1.getPoint()
        self._x2, self._y2 = obj2.getPoint()
    
    def __str__(self):
        result = self._calculate_distance()
        return str(result)

    def getDistance(self):
        return self._calculate_distance()

    def setDistance(self, obj1:Point, obj2: Point):
        self._x1, self._y1 = obj1.getPoint()
        self._x2, self._y2 = obj2.getPoint()
        
    def _calculate_distance(self):
        x = math.pow(self._x2 - self._x1, 2)
        y = math.pow(self._y2 - self._y1, 2)
        return math.sqrt(x+y)

class Shape:

    def __init__(self, shape):
        self._shape = shape
    
    def __str__(self):
        return self._shape

    def getShape(self):
        return self._shape
    
    def setShape(self, shape):
        self._shape = shape
    
    def area(self):
        pass
    
    def draw(self):
        pass

    def resize(self, factor):
        pass

    def getEquation(self):
        pass

class CircleRadiusForm(Shape):
    # Contains all functions of circle

    def __init__(self, r, origin:Point):
        Shape.__init__(self, "Circle")
        self._shape = self.getShape()
        self._x, self._y = origin.getPoint()
        self._radius = r

    def __str__(self):
        return str(self.getEquation())
    
    def area(self):
        return math.pi*(self._radius**2)

    def getCircle(self):
        return self._radius, Point(self._x, self._y)
    
    def setCircle(self, radius, origin:Point):
        shape.__init__(self, "Circle")
        self._shape = self.getShape()
        self._x, self._y = origin.getPoint()
        self._radius = radius

    def draw(self):
        angle = np.linspace(0, 2*np.pi, 300)
        radius = self._radius

        x = self._x + (radius * np.cos(angle))
        y = self._y + (radius * np.sin(angle))

        axes = plt.gca()
        axes.set_aspect(1, adjustable= 'datalim')

        axes.plot(x,y)

        plt.title('Circle')
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')

        plt.show()

    def resize(self, factor):
        self._radius *= factor

    def getEquation(self):
        a = 1
        b = 1
        r = self._radius
        g = self._x
        f = self._y
        c = (g**2 + f**2 - r**2)
        sympy.init_printing()
        y,x = sympy.symbols(('y', 'x'))

        return sympy.Eq(x**2+y**2+2*g*x+2*f*y+c, 0)

class TwoDegreeEquation(Shape):
    def __init__(self, *args):
        if len(args) == 6:
            self.a = args[0]
            self.h = args[1]
            self.b = args[2]
            self.g = args[3]
            self.f = args[4]
            self.c = args[5]
            self.arg = args
        else:
            raise TypeError("Consider the general two degree equation as ax^2+2hxy+by^2+2gx+2fy+x = 0 \n\n \
            TwoDegreeEquation takes only 6 args - inorder as a,h,b,g,f,c")

    def setValue(self, *args):
        self.__init__(*args)
    
    def __str__(self):
        return str(self.getEquation())
    
    def getEquation(self):

        y,x = sympy.symbols(('y','x'))
        sympy.init_printing()

        return sympy.Eq((self.a*(x**2)) + (2*self.h*x*y) + (self.b*(y**2)) + (2*self.g*x) + (2*self.f*y) + (self.c), 0)

    def checkShape(self):

        k = (self.a*self.b*self.c) + (2*self.f*self.g*self.h) - (self.a*(self.f**2)) - (self.b*(self.g**2)) - (self.c*(self.h**2))

        if round(k) == 0:
            return "PairOfStraightLine"
        elif self.a == self.b and self.h == 0 and self.g**2+self.f**2-self.c > 0:
            return "Circle"
        elif self.a == self.b and self.h == 0 and self.g**2+self.f**2-self.c == 0:
            return "Point"
        elif self.h**2 == self.a*self.b:
            return "Parabola"
        elif self.h**2 < self.a*self.b:
            return "Ellipse"
        elif self.h**2 > self.a*self.b:
            return "Hyperbola"
        else: return

    def draw(self):
        x = np.linspace(-1000,1000,300)
        y = np.linspace(-1000,1000,300)
        X,Y = np.meshgrid(x,y)
        F = self.a*X**2 + 2*self.h*X*Y + self.b*Y**2 + 2*self.g*X + 2*self.f*Y + self.c

        fig, ax = plt.subplots()
        ax.contour(X,Y,F,levels=[0])
        plt.show()

    def isPointOnCurve(self,X,Y):
        F = self.a*X**2 + 2*self.h*X*Y + self.b*Y**2 + 2*self.g*X + 2*self.h*Y + self.c
        if F == 0:
            return True
        else: False
    
    def isPointInside(self,X,Y):
        raise NotImplementedError(f"isPointInside() is not yet implemented to {self.checkShape()}")
    
    def isPointOutside(self,x,Y):
        raise NotImplementedError(f"isPointOutside() is not yet implemented yet to {self.checkShape()}")

class PairOfStraightLine(TwoDegreeEquation):
    def __init__(self, *args):
        if len(args) == 6:
            TwoDegreeEquation.__init__(self, *args)
            if self.getShape() != "PairOfStraightLine":
                raise ValueError("Not a Pair of Straight Line")
            else:
                raise TypeError("Consider the general two degree equation as ax^2+2hxy+by^2+2gx+2fy+x = 0 \n\n \
            TwoDegreeEquation takes only 6 args - inorder as a,h,b,g,f,c")

    def setValue(self, *args):
        self.__init__(*args)
    
    def __str__(self):
        return str(self.getEquation())

    def getEquation(self):

        y,x = sympy.symbols(('y','x'))

        sympy.init_printing()
        return sympy.Eq((self.a*(x**2)) + (2*self.h*x*y) + (self.b*(y**2)) + (2*self.g*x) + (2*self.h*y) + self.c, 0)
    
    def draw(self):
        x = np.linspace(-1000,1000,300)
        y = np.linspace(-1000,1000,300)

        X, Y = np.meshgrid(x,y)
        F = self.a*X**2 + 2*self.h*X*Y + self.b*Y**2 + 2*self.g*X + 2*self.f*Y + self.c
        fig, ax = plt.subplots()
        ax.contour(X,Y,F, levels = [0])
        plt.show()

    def getSlopes(self):
        m1 = sympy.symbols('m1')
        expression = sympy.Eq((self.b**2)*(m1**2) + (2*self.h*self.b*m1) + self.a, 0)
        a = sympy.solve(expression, m1)

        m1,m2 = a
        
        return sympy.N(m1), sympy.N(m2)

    def getLines(self):
        a1, a2 = self.getSlopes()
        m, n = sympy.symbols('m n')
        e1 = sympy.Eq(-(a1*m + a2*n), 2*self.g)
        e2 = sympy.Eq((m+n), 2*self.f)
        sol = sympy.solve((e1,e2), (m,n))
        c1 = sympy.N(sol[m]); c2 = sympy.N(sol[n])
        print(c1*c2 == self.c)
        x,y = sympy.symbols('x y')
        l1 = sympy.Eq(y-a1*x+c1, 0)
        l2 = sympy.Eq(y-a2*x+c2, 0)
        return l1,l2
    
    def getAngle(self):
        if self.a + self.b == 0:
            theta = 90
        else:
            theta = math.degrees(math.tanh(2*math.sqrt(self.h**2 - self.a*self.b)/(self.a+self.b)))
        return theta

    def getPointOfIntersection(self):
        x,y = sympy.symbols('x y')
        expression = self.a*(x**2) + (2*self.h*x*y) + self.b*(y**2) + 2*self.g*x + 2*self.f*y + self.c
        l1 = sympy.Eq(sympy.diff(expression, y), 0)
        l2 = sympy.Eq(sympy.diff(expression,x),0)
        sol = sympy.solve((l1,l2), (x,y))
        return Point(sol[x], sol[y])
    
    def getFamilyOfLines(self):
        a = self.getPointOfIntersection()
        x1,y1 = a.getPoint()
        m1,m2 = self.getSlopes()
        x,y,m = sympy.symbols('x y m')
        l = sympy.Eq(y - m*x + m1*x1 - y1, 0)
        
        return l
    
    def getAngleBisectors(self):
        m1,m2 = self.getSlopes()
        a = self.getPointOfIntersection()
        x1,y1 = getPoint()
        x,y,m = sympy.symbols('x y m')
        e1 = ((m2-m)/1+m2*m)
        e2 = ((m1-m)/1+m1*m)
        exp = sympy.Eq(e1,e2)
        sol1 = sympy.solve(exp, m)

        sol1 = sol1[0]
        sol2 = -1.0/sol1

        l1 = sympy.Eq(y-sol1*x+sol1*x1-y1, 0)
        l2 = sympy.Eq(y-sol2*x+sol2*x1-y1, 0)

        return l1,l2


class StraightLine(Shape):
    def __init__(self, a, b,c):
            self.a = a
            self.b = b
            self.c = c

            try:
                self.slope = -a/b
            except ZeroDivisionError:
                self.slope = math.pow(10,10)
            Shape.__init__(self, "Straight Line")
        
    def __str__(self):
            return str(self.getEquation())
        
    def getEquation(self):
            x,y = sympy.symbols("x y")
            return sympy.Eq(self.a*x + self.b * y + self.c,0)
        
    def getSlope(self):
            return self.slope
        
    def draw(self):
            x = np.linspace(-1000,1000,300)
            y = np.linspace(-1000,1000,300)
            X,Y = np.meshgrid(x,y)
            F = self.a*X+self.b*Y+self.c

            fig,ax = plt.subplots()
            ax.contour(X,Y,F,levels=[0])
            plt.show()

    def getAngle(self, line2):
            m1 = line2.getSlope()
            m2 = self.getSlope()

            if m1*m2 != 1:
                a = abs((m2-m1)/(1+m1*m2))
                theta = math.degrees(math.tanh(a))
            else: return 90
            return theta
        
    def isPointOnLine(self,X,Y):
            F = self.a*X + self.b*Y + self.c
            if F == 0: return True
            else: return False

    def distanceOfPointFromLine(self, point):
            x,y = point.getPoint()
            a,b,c = self.a, self.b, self.c
            d = abs(a*x+b*y+c)/math.sqrt(a**2+b**2)
            return d

    def footOfPointOnLine(self, point):
            x,y = point.getPoint()
            a,b,c = self.a, self.b, self.c
            d = -1*(abs(a*x+b*y+c)/(a**2+b**2))
            x1 = x+a*d
            y1 = y+b*d
            return Point(x1, y1)

    def imageOfPointInLine(self, point):
            x,y = point.getPoint()
            a,b,c = self.a,self.b,self.c
            d = -2*(abs(a*x+b*y+c)/(a**2+b**2))
            x1 = x+a*d
            y1 = y + b*d
            return Point(x1,y1)
        
    def getPointOfIntersection(self, line):
            x,y = sympy.symbols('x y')
            l1 = self.a*x+self.b*y+self.c
            l2 = line.a*x+line.b*y+line.c
            sol = sympy.solve((l1,l2),(x,y))
            return Point(sol[x], sol[y])

    def getParallelLine(self, point):
            slope = self.getSlope()
            return getLinePointSlope(point, slope)
            

    def getPerpendicularLine(self, point):
            X,Y = point.getPoint()
            slope = self.getSlope()
            x = sympy.symbols('x')
            if slope != 0 : 
                slope = -1/slope
                return getLinePointSlope(point, slope)
            else:
                return sympy.Eq(x-X, 0)

    def getFamilyOfLines(self, other):
            P = self.getPointOfIntersection(other)
            x1,y1 = P.getPoint()
            x,y,m = sympy.symbols('x y m')
            e = sympy.Eq((y-y1) - m*(x-x1), 0)
            return e

class Circle(TwoDegreeEquation):
    def __init__(self, *args):

        if len(args) == 6:
            TwoDegreeEquation.__init__(self, *args)
            if self.getShape() != "Circle":
                    raise ValueError("Not a Circle")
            if self.a != 0:
                    self.g/=self.a
                    self.f/=self.a
                    self.c/=self.a
                    self.b/=self.a
                    self.a /= self.a
            self.radius = math.sqrt(self.g**2+self.f**2-self.c)
        else:
                raise TypeError("Enter only 6 args in order as a,h,b,g,f,c")

    def setValue(self,*args):
        self.__init__(*args)
    
    def __str__(self):
        return str(self.getEquation())

    def getEquation(self):
        y,x = sympy.symbols('y x')
        sympy.init_printing()
        return sympy.Eq(self.a*(x**2)+2*self.h*x*y+self.b*(y**2)+2*self.g*x+2*self.f*y+self.c, 0)
    
    def getRadius(self):
        return self.radius

    def getCenter(self):
        return Point(-self.g,-self.f)

    def draw(self):
        angle = np.linspace(0,2*np.pi,300)
        radius = self.radius
        x = -self.g + (radius*np.cos(angle))
        y = -self.f + (radius*np.sin(angle))

        axes = plt.gca()
        axes.set_aspect(1,adjustable="datalim")
        axes.plot(x,y)

        plt.title('Circle')
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')

        plt.show()

    def area(self):
        r = self.radius
        return np.pi*(r**2)
    
    def resize(self, factor):
        self.radius *= (factor)
    
    def FamilyOfCircles(self,other):

        x,y,b = sympy.symbols('x y b')
        c1 = self.a*(x**2) + 2*self.h*x*y + self.b*(y**2) + 2*self.g*x + 2*self.f*y + self.c
        c2 = other.a*(x**2) + 2*other.h*x*y + other.b*(y**2) + 2*other.g*x + 2*other.f*y + other.c
        
        exp = c1 + b*c2
        return sympy.Eq(exp, 0)
    
    def getChordLength(self,X,Y):
        '''X and Y are coordinates of the midpoint of the chord'''
        if self.isPointInside(X,Y):
            Po = self.getCenter()
            r = self.radius
            P1 = Point(X,Y)
            d = (distance(Po,P1))
            d = str(d)
            d = float(d)
            L = 2*math.sqrt(r**2-d**2)
            return L

    def isPointInside(self,X,Y):
        F = self.a*X**2 + 2*self.h*X*Y + self.b*Y**2 + 2*self.g*X + 2*self.h*Y + self.c
        return True if F < 0 else False
    
    def isPointOutside(self,X,Y):
        F = self.a*X**2 + 2*self.h*X*Y + self.b*Y**2 + 2*self.g*X + 2*self.h*Y + self.c
        return True if F > 0 else False

    def EquationOfChord(self,X,Y):
        y,x = sympy.symbols(('y','x'))
        S1 = self.a*X**2 + 2*self.h*X*Y + self.b*Y**2 + 2*self.g*X + 2*self.f*Y + self.c
        T = self.a*X*x + self.h*x*Y + self.h*y*X + self.b*Y*y + self.g*(X+x) + self.f*(Y+y)+self.c
        f = T-S1
        return sympy.Eq(T-S1, 0)

    def RadicalAxis(self,other):
        x,y = sympy.symbols('x y')
        s1 = self.a*x**2+2*self.h*x*y+self.b*y**2+2*self.g*x+2*self.f*y+self.c
        s2 = other.a*x**2+2*other.h*x*y+other.b*y**2+other.g*x+2*other.f*y+other.c

        return sympy.Eq(s1-s2)



# Testing Code

pt1 = Point(4,0)
# pt.setPoint(-100,-123123)
# a = pt.getPoint()
# print(a)
pt2 = Point(0,4)
# print(pt1.midPoint(pt))
# print(pt1.sectionPoint(pt,1,3))
# print(pt1.getLinePointSlope(2))
# print(Distance(pt1, pt2))
# d = Distance(pt1, pt2)
# print(d.getDistance())
origin = Point(5,-5)
circle = CircleRadiusForm(10, origin)
print(circle.getEquation())
