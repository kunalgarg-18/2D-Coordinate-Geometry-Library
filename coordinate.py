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

class Triangle(Shape):
    def __init__(self, l1: StraightLine, l2: StraightLine, l3: StraightLine):
        self.line1 = l1
        self.line2 = l2
        self.line3 = l3
        self.point1 = l1.getPointOfIntersection(l2)
        self.point2 = l2.getPointOfIntersection(l3)
        self.point3 = l3.getPointOfIntersection(l1)
        self.angle1 = l1.getAngle(l2)
        self.angle2 = l2.getAngle(l3)
        self.angle3 = l3.getAngle(l1)
        self.slope1 = l1.getSlope()
        self.slope2 = l2.getSlope()
        self.slope3 = l3.getSlope()
        Shape.__init__(self, "Triangle")

        if not self.isTriangle():
            raise ValueError("This is not a triangle!")
        
    def setTriangle(self,l1,l2,l3):
        self.__init__(l1,l2,l3)
    
    def getTriangle(self):
        return self.line1, self.line2, self.line3

    def isTriangle(self):
        a = self.C = float(str(Distance(self.point1, self.point2)))
        b = self.A = float(str(Distance(self.point2, self.point3)))
        c = self.B = float(str(Distance(self.point3, self.point1)))
        if a+b > c or b+c > a or a+c > b:
            return True
        return False
    
    def isAcuteAngleTriangle(self):
        if self.angle1 < 90 and self.angle2 < 90 and self.angle3 < 90:
            return True
        return False
    
    def isRightAngleTriangle(self):
        if self.angle1 == 90 or self.angle2 == 90 or self.angle3 == 90:
            return True   
        return False
    
    def isObtuseAngleTriangle(self):
        if self.angle1 > 90 or self.angle2 > 90 or self.angle3 > 90:
            return True
        return False

    def getTrianglePoints(self):
        return self.point1, self.point2, self.point3
    
    def getTriangleAngles(self):
        return self.angle1, self.angle2, self.angle3

    def getTriangleSides(self):
        return self.A, self.B, self.C
    
    def getTriangleSlopes(self):
        return self.slope1, self.slope2, self.slope3

    def isEquilateralTriangle(self):
        if self.A == self.B == self.C == 0 or self.angle1 == self.angle2 == self.angle3 == 60:
            return True
        return False

    def isIsoscelesTriangle(self):
        if self.A == self.B or self.B == self.C or self.C == self.A:
            return True
        return False
    
    def isScaleneTriangle(self):
        if self.A != self.B != self.C or self.angle1 != self.angle2 != self.angle3:
            return True
        return False

    def area(self):
        a,b,c = self.A, self.B, self.C
        s = (a+b+c)/2
        return math.sqrt(s*(s-a)*(s-b)*(s-c))

    def draw(self):
        l1,l2,l3 = self.getTriangle()
        a1,b1,c1 = l1.a, l1.b,l1.c
        a2,b2,c2 = l2.a, l2.b, l2.c
        a3,b3,c3 = l3.a, l3.b, l3.c
        x = np.linspace(-100,100,300)
        y = np.linspace(-100,100,300)
        X,Y = np.meshgrid(x,y)
        F1 = a1*X+b1*Y+c1
        F2 = a2*X+b2*Y+c2
        F3 = a3*x+b3*y+c3

        fig,ax = plt.subplots()
        ax.contour(X,Y,F1,levels=[0])
        ax.contour(X,Y,F2, levels=[0])
        ax.contour(X,Y,F3,levels=[0])
        plt.show()

    def getCircumradius(self):
        c = self.A*self.B*self.C / (4*self.area())
        return r

    def getInradius(self):
        s = (self.A+self.B+self.C)/2
        r = self.area()/s
        return r
    
    def getExradius(self):
        s = (self.A+self.B+self.C)/2
        a = self.area()
        r1 = a/(s-self.A)
        r2 = a/(s-self.B)
        r3 = a/(s-self.C)
        return r1,r2,r3
    
    def getAngleBisectorLength1(self):
        a,b,c = self.A,self.B,self.C
        s = (a+b+c)/2
        cosAby2 = math.sqrt((s*(s-a))/(b*c))
        return (2*b*c*cosAby2)/(b+c)
    
    def getAngleBisectorLength2(self):
        a,b,c = self.A,self.B,self.C
        s = (a+b+c)/2
        cosBby2 = math.sqrt((s*(s-b))/(a*c))
        return (2*a*c*cosBby2)/(a+c)
    
    def getAngleBisectorLength3(self):
        a,b,c = self.A,self.B,self.C
        s = (a+b+c)/2
        cosCby2 = math.sqrt((s*(s-c))/(b*a))
        return (2*b*a*cosCby2)/(b+a)
    
    def getMedianLength1(self):
        a,b,c = self.A,self.B,self.C
        s = (a+b+c)/2

        return 0.5*math.sqrt(2*(b**2)+2*(c**2)-a**2)

    def getMedianLength2(self):
        a,b,c = self.A,self.B,self.C
        s = (a+b+c)/2

        return 0.5*math.sqrt(2*(a**2)+2*(c**2)-b**2)

    def getMedianLength3(self):
        a,b,c = self.A,self.B,self.C
        s = (a+b+c)/2

        return 0.5*math.sqrt(2*(a**2)+2*(b**2)-c**2)

    def getAltitudeLength1(self):
        a,b,c = self.A,self.B,self.C
        A = self.area()
        
        return 2*A/a

    def getAltitudeLength2(self):
        a,b,c = self.A,self.B,self.C
        A = self.area()
        
        return 2*A/b
    
    def getAltitudeLength3(self):
        a,b,c = self.A,self.B,self.C
        A = self.area()
        
        return 2*A/c

    def sinA(self):
        R = self.getCircumradius()
        return self.A/R

    def sinB(self):
        R = self.getCircumradius()
        return self.B/R
    
    def sinC(self):
        R = self.getCircumradius()
        return self.C/R

    def sinAby2(self):
        a,b,c = self.A,self.B,self.C
        s = (a+b+c)/2
        
        return math.sqrt(((s-b)*(s-c))/(b*c))
    
    def sinBby2(self):
        a,b,c = self.A,self.B,self.C
        s = (a+b+c)/2
        
        return math.sqrt(((s-a)*(s-c))/(a*c))
    
    def sinCby2(self):
        a,b,c = self.A,self.B,self.C
        s = (a+b+c)/2
        
        return math.sqrt(((s-b)*(s-a))/(b*a))

    def cosA(self):
        a,b,c = self.A,self.B,self.C
        return ((b**2)+(c**2)-(a**2))/(2*b*c)

    def cosB(self):
        a,b,c = self.A,self.B,self.C
        return ((a**2)+(c**2)-(b**2))/(2*a*c)

    def cosC(self):
        a,b,c = self.A,self.B,self.C
        return ((b**2)+(a**2)-(c**2))/(2*b*a)
    
    def cosAby2(self):
        a,b,c = self.A,self.B,self.C
        s = (a+b+c)/2
        return math.sqrt((s*(s-a))/(b*c))

    def cosBby2(self):
        a,b,c = self.A,self.B,self.C
        s = (a+b+c)/2
        return math.sqrt((s*(s-b))/(a*c))
    
    def cosCby2(self):
        a,b,c = self.A,self.B,self.C
        s = (a+b+c)/2
        return math.sqrt((s*(s-c))/(b*a))

    def tanAby2(self):
        return self.sinAby2/self.cosAby2
    
    def tanBby2(self):
        return self.sinBby2/self.cosBby2
    
    def tanCby2(self):
        return self.sinCby2/self.cosCby2
    
    def isPointInsideTriangle(self, point: Point):
        x,y = point.getPoint()

        l1,l2,l3 = self.getTriangle()

        a1,b1,c1 = l1.a,l1.b,l1.c
        a2,b2,c2 = l2.a,l2.b,l2.c
        a3,b3,c3 = l3.a,l3.b,l3.c

        if b1 < 0:
            a1,b1,c1 = -a1,-b1,-c1
        if b2 < 0:
            a2,b2,c2 = -a2,-b2,-c2
        if b3 < 0:
            a3,b3,c3 = -a3,-b3,-c3
        
        F = (a1*x+b1*y+c1)*(a2*x+b2*y+c2)*(a3*x+b3*y+c3)
        if F < 0: return True
        else: return False

    def isPointOutsideTriangle(self, point: Point):
        x,y = point.getPoint()

        l1,l2,l3 = self.getTriangle()

        a1,b1,c1 = l1.a,l1.b,l1.c
        a2,b2,c2 = l2.a,l2.b,l2.c
        a3,b3,c3 = l3.a,l3.b,l3.c

        if b1 < 0:
            a1,b1,c1 = -a1,-b1,-c1
        if b2 < 0:
            a2,b2,c2 = -a2,-b2,-c2
        if b3 < 0:
            a3,b3,c3 = -a3,-b3,-c3
        
        F = (a1*x+b1*y+c1)*(a2*x+b2*y+c2)*(a3*x+b3*y+c3)
        if F > 0: return True
        else: return False


    def getCircumcenter(self):
        p1,p2,p3 = self.getTrianglePoints()
        x1,y1 = p1.getPoint()
        x2,y2 = p2.getPoint()
        x3,y3 = p3.getPoint()

        sinA,sinB,sinC = self.sinA(), self.sinB(), self.sinC()
        cosA,cosB,cosC = self.cosA(), self.cosB(), self.cosC()
        sin2A,sin2B,sin2C = 2*sinA*cosA, 2*sinB*cosB, 2*sinC*cosC

        x = ((x1*sin2A)+(x2*sin2B)+(x3*sin2C))/(sin2A+sin2B+sin2C)
        y = ((y1*sin2A)+(y2*sin2B)+(y3*sin2C))/(sin2A+sin2B+sin2C)

        return Point(x,y)

    def getIncenter(self):
        p1,p2,p3 = self.getTrianglePoints()
        x1,y1 = p1.getPoint()
        x2,y2 = p2.getPoint()
        x3,y3 = p3.getPoint()

        a,b,c = self.A,self.B,self.C

        x = (a*x1+b*x2+c*x3)/(a+b+c)
        y = (a*y1+b*y2+c*y3)/(a+b+c)

        return Point(x,y)

    def getExcenters(self):
        p1,p2,p3 = self.getTrianglePoints()
        x1,y1 = p1.getPoint()
        x2,y2 = p2.getPoint()
        x3,y3 = p3.getPoint()

        a,b,c = self.A,self.B,self.C

        ix1 = (-a*x1+b*x2+c*x3)/(-a+b+c)
        iy1 = (-a*y1+b*y2+c*y3)/(-a+b+c)
        ix2 = (a*x1-b*x2+c*x3)/(a-b+c)
        iy2 = (a*y1-b*y2+c*y3)/(a-b+c)
        ix3 = (a*x1+b*x2-c*x3)/(a+b-c)
        iy3 = (a*y1+b*y2-c*y3)/(a+b-c)

        return Point(ix1,iy1), Point(ix2,iy2), Point(ix3, iy3)

    def getExcentralTriangle(self):
        R = self.getCircumcenterLength()
        A = 4*R*self.cosAby2()
        B = 4*r*self.cosBby2()
        C = 4*r*self.cosCby2()

        return A,B,C

    def getOrthocenter(self):
        p1,p2,p3 = self.getTrianglePoints()
        x1,y1 = p1.getPoint()
        x2,y2 = p2.getPoint()
        x3,y3 = p3.getPoint()
        
        tanA2,tanB2,tanC2 = self.tanAby2(),self.tanBby2(),self.tanCby2()
        tanA = (2*tanA2)/(1-(tanA2**2))
        tanB = (2*tanB2)/(1-(tanB2**2))
        tanC = (2*tanC2)/(1-(tanC2**2))

        x = ((x1*tanA)+(x2*tanB)+(x3*tanC))/(tanA+tanB+tanC) 
        y = (y1*tanA + y2*tanB + y3*tanC)/(tanA+tanB+tanC)
        
        return Point(x,y)

    def getCentroid(self):
        p1,p2,p3 = self.getTrianglePoints()
        x1,y1 = p1.getPoint()
        x2,y2 = p2.getPoint()
        x3,y3 = p3.getPoint()

        x = (x1+x2+x3)/3
        y = (y1+y2+y3)/3

        return Point(x,y)

    def distanceBetweenCircumcenterOrthocenter(self):
        Pc=self.getCicumcenter()
        Po=self.getOrthocenter()
        d=float(str(Distance(Pc,Po)))
        return d

    def distanceBetweenCircumcenterIncenter(self):
        Pc=self.getCicumcenter()
        Pi=self.getIncenter()
        d=float(str(Distance(Pc,Pi)))
        return d

    def distanceBetweenCircumcenterCentroid(self):
        Pc=self.getCicumcenter()
        Pg=self.getCentroid()
        d=float(str(Distance(Pc,Pg)))
        return d

    def distanceBetweenIncenterOrthocenter(self):
        Pi=self.getIncenter()
        Po=self.getOrthocenter()
        d=float(str(Distance(Pi,Po)))
        return d

    def distanceBetweenCentroidOrthocenter(self):
        Pg=self.getCentroid()
        Po=self.getOrthocenter()
        d=float(str(Distance(Pg,Po)))
        return d
class Quadrilateral(Shape):
    def __init__(self, l1:StraightLine, l2:StraightLine,l3:StraightLine,l4:StraightLine):
        self.line1 = l1
        self.line2 = l2
        self.line3 = l3
        self.line4 = l4

        self.point1 = l1.getPointOfIntersection(l2)
        self.point2 = l2.getPointOfIntersection(l3)
        self.point3 = l3.getPointOfIntersection(l4)
        self.point4 = l4.getPointOfIntersection(l1)

        self.side1 = float(str(Distance(self.point1, self.point2)))
        self.side2 = float(str(Distance(self.point2, self.point3)))
        self.side3 = float(str(Distance(self.point3, self.point4)))
        self.side4 = float(str(Distance(self.point4, self.point1)))

        self.angle1 = l1.getAngle(l2)
        self.angle2 = l2.getAngle(l3)
        self.angle3 = l3.getAngle(l4)
        self.angle4 = l4.getAngle(l1)

        self.slope1 = l1.getSlope()
        self.slope2 = l2.getSlope()
        self.slope3 = l3.getSlope()
        self.slope4 = l4.getSlope()

        self.shape = self.checkShape()
        Shape.__init__(self,self.shape)

        def setQuadrilateral(self,l1,l2,l3,l4):
            self.__init__(l1,l2,l3,l4)

        def __str__(self):
            return str(self.getLineEquations())

        def getQuadrilateral(self):
            return self.line1,self.line2,self.line3,self.line4

        def getSideLength(self):
            return self.side1, self.side2, self.side3, self.side4

        def getSlopes(self):
            return self.slope1, self.slope2, self.slope3, self.slope4
        
        def getAngles(self):
            return self.angle1, self.angle2, self.angle3, self.angle4
        
        def getPoints(self):
            return self.point1, self.point2, self.point3, self.point4
        
        def getCenter(self):
            x, y = sympy.symbols('x y')
            d1,d2 = self.getDiagonalEquations()
            sol = sympy.solve((d1,d2),(x,y))
            return Point(sol[x],sol[y])
        
        def draw(self):
            l1,l2,l3,l4 = self.getSquare()
            a1,b1,c1 = l1.a,l1.b,l1.c
            a2,b2,c2 = l2.a,l2.b,l2.c
            a3,b3,c3 = l3.a,l3.b,l3.c
            a4,b4,c4 = l4.a,l4.b,l4.c

            x = np.linspace(-100,100,300)
            y = np.linspace(-100,100,300)
            X,Y = np.meshgrid(x,y)
            
            F1 = a1*X+b1*Y+c1
            F2 = a2*X+b2*Y+c2
            F3 = a3*X+b3*Y+c3
            F4 = a4*X+b4*Y+c4

            fig,ax = plt.subplots()
            ax.contour(X,Y,F1,levels=[0])
            ax.contour(X,Y,F2, levels = [0])
            ax.contour(X,Y,F3, levels = [0])
            ax.contour(X,Y,F4,levels=[0])
            plt.show()

        def area(self):
            pass

        def getDiagonalLength(self):
            a,b,c,d = self.getPoints()
            d1 = float(str(Distance(a,c)))
            d2 = float(str(Distance(b,d)))
            return d1,d2

        def getDiagonalEquations(self):
            A,B,C,D = self.getPoints()
            ax1,ay1 = A.getPoint()
            bx1,by1 = B.getPoint()
            cx1,cy1 = C.getPoint()
            dx1,dy1 = D.getPoint()

            m1 = (cy1-ay1)/(cx1-ax1)
            m2 = (dy1-by1)/(dx1-bx1)

            x,y = sympy.symbols('x y')
            d1 = sympy.Eq((y-ay1)-m1*(x-ax1),0)
            d2 = sympy.Eq((y-by1)-m2*(x-bx1),0)

            return d1,d2
        
        def getLineEquations(self):
            return self.getQuadrilateral()
        
        def getPerimeter(self):
            a,b,c,d = self.getSideLength()
            return a+b+c+d
        
        def checkShape(self):
            a,b,c,d = self.getSideLength()
            A,B,C,D = self.getAngles()

            if a==b==c==d and A==B==C==D==90:
                return "Square"
            elif a==b==c==d and A==C and B==D:
                return "Rhombus"
            elif a==c and b==d and A==B==C==D==90:
                return "Rectangle"
            elif a==c and b==d and A==C and B==D:
                return "Parallelogram"
            elif A+C == 180 and B+D == 180:
                return "Cyclic Quadrilateral"
            elif (A+B == 180 and C+D == 180) or (A+D == 180 and B+C == 180):
                return "Trapezium"
            else: return "Quadrilateral"
        
        def getAngleBisectors(self):
            pass

        def getDiagonalAngles(self):
            d1,d2 = self.getDiagonalEquations()
            angle1 = d1.getAngle(d2)
            angle2 = 180-angle1

            return angle1,angle2

class Square(Quadrilateral):
    def __init__(self, l1:StraightLine,l2:StraightLine,l3:StraightLine,l4:StraightLine):
        Quadrilateral.__init__(self,l1,l2,l3,l4)
        if self.shape != "Square":
            raise ValueError("The given lines do not form a Square")
        self.center = self.getCenter()

    def getCenter(self):
        a = self.point1
        b = self.point3
        return self.point1.midPoint(self.point3)
    
    def area(self):
        a,b,c,d = self.getSideLength()
        return a*b
    
    def getAngleBisectorsEquation(self):
        return self.getDiagonalEquations()
    
    def getInradiusLength(self):
        a = self.getDiagonalLength()
        return a/(2*math.sqrt(2))
    
    def getCircumradiusLength(self):
        a = self.getDiagonalLength()
        return a/2
    
    def isPointInsideSquare(self, point:Point):
        x,y = point.getPoint()
        p1,p2,p3,p4 = self.getPoints()
        x1,y1 = p1.getPoint()
        x2,y2 = p2.getPoint()
        x3,y3 = p3.getPoint()
        x4,y4 = p4.getPoint()

        def area(x1,y1,x2,y2,x3,y3):

            return abs((x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))/2.0)

        A1 = area(x1,y1,x2,y2,x,y)
        A2 = area(x2,y2,x3,y3,x,y)
        A3 = area(x3,y3,x4,y4,x,y)
        A4 = area(x4,y4,x1,y1,x,y)

        A = area(x1,y1,x2,y2,x3,y3) + area(x1,y1,x3,y3,x4,y4)

        return A == (A1+A2+A3+A4)
    
    def isPointOutsideSquare(self):
        return not self.isPointInsideSquare()

class Rectangle(Quadilateral):
    def __init__(self, l1: StraightLine, l2: StraightLine, l3: StraightLine, l4: StraightLine):
        super().__init__(l1, l2, l3, l4)
        if self.shape != "Rectangle":
            raise ValueError("The given lines do not form a Rectangle")
        self.center=self.getCenter()
    
    def getCenter(self):
        a=self.point1
        b=self.point3
        c=self.point1.findMidpoint(self.point3)
        return c

    def getAngleBisectorsEquations(self):
        return self.getDiagonalEquations()

    def getCircumradiusLength(self):   
        a=self.getDiagonalLength()
        R=a/2
        return R

    def area(self):
        a,b,c,d=self.getSideLength()
        return a*b

    def isGoldenRectangle(self):
        a,b=self.getSideLength()
        l= a*b + b*b
        k=a*a
        if l==k:
            return True
        return False

    def isPointPresentInside(self,point:Point):
        x,y=point.getXYloc()
        p1,p2,p3,p4=self.getPoints()
        x1,y1=p1.getXYloc()
        x2,y2=p2.getXYloc()
        x3,y3=p3.getXYloc()
        x4,y4=p4.getXYloc()

        def area(x1, y1, x2, y2, x3, y3):
            return abs((x1 * (y2 - y3) + 
                    x2 * (y3 - y1) + 
                    x3 * (y1 - y2)) / 2.0)

        A1= area(x1,y1,x2,y2,x,y)
        A2= area(x3,y3,x2,y2,x,y)
        
        A3= area(x3,y3,x4,y4,x,y)
        A4= area(x1,y1,x4,y4,x,y)
        
        A = (area(x1, y1, x2, y2, x3, y3) + area(x1, y1, x4, y4, x3, y3))
        
        return A==(A1+A2+A3+A4)

    def isPointPresentOutside(self):
        return not self.isPointPresentInside()

class Rhombus(Quadilateral):
    def __init__(self, l1: StraightLine, l2: StraightLine, l3: StraightLine, l4: StraightLine):
        super().__init__(l1, l2, l3, l4)
        if self.shape != "Rectangle":
            raise ValueError("The given lines do not form a Rectangle")
        self.center=self.getCenter()
    
    def getCenter(self):
        a=self.point1
        b=self.point3
        c=self.point1.findMidpoint(self.point3)
        return c

    def getAngleBisectorsEquations(self):
        return self.getDiagonalEquations()

    def getCircumradiusLength(self):   
        a=self.getDiagonalLength()
        R=a/2
        return R

    def area(self):
        d1,d2=self.getDiagonalLength()
        return d1*d2
    
    def isPointPresentInside(self,point:Point):
        x,y=point.getXYloc()
        p1,p2,p3,p4=self.getPoints()
        x1,y1=p1.getXYloc()
        x2,y2=p2.getXYloc()
        x3,y3=p3.getXYloc()
        x4,y4=p4.getXYloc()
        def area(x1, y1, x2, y2, x3, y3):
            return abs((x1 * (y2 - y3) + 
                    x2 * (y3 - y1) + 
                    x3 * (y1 - y2)) / 2.0)

        A1= area(x1,y1,x2,y2,x,y)
        A2= area(x3,y3,x2,y2,x,y)
        
        A3= area(x3,y3,x4,y4,x,y)
        A4= area(x1,y1,x4,y4,x,y)
        
        A = (area(x1, y1, x2, y2, x3, y3) + area(x1, y1, x4, y4, x3, y3))
        
        return A==(A1+A2+A3+A4)

    def isPointPresentOutside(self):
        return not self.isPointPresentInside()
    

class Parallelogram(Quadilateral):
    def __init__(self, l1: StraightLine, l2: StraightLine, l3: StraightLine, l4: StraightLine):
        super().__init__(l1, l2, l3, l4)
        if self.shape != "Rectangle":
            raise ValueError("The given lines do not form a Rectangle")
        self.center=self.getCenter()
    
    def getCenter(self):
        a=self.point1
        b=self.point3
        c=self.point1.findMidpoint(self.point3)
        return c

    def getAngleBisectorsEquations(self):
        return self.getDiagonalEquations()

    def getCircumradiusLength(self):   
        a=self.getDiagonalLength()
        R=a/2
        return R

    def area(self):
        d1,d2=self.getDiagonalLength()
        a1,a2=self.getDiagonalAngles()
        r=(math.pi*a1)/180
        return d1*d2* math.sin(r)
    
    def isPointPresentInside(self,point:Point):
        x,y=point.getXYloc()
        p1,p2,p3,p4=self.getPoints()
        x1,y1=p1.getXYloc()
        x2,y2=p2.getXYloc()
        x3,y3=p3.getXYloc()
        x4,y4=p4.getXYloc()
        def area(x1, y1, x2, y2, x3, y3):
            return abs((x1 * (y2 - y3) + 
                    x2 * (y3 - y1) + 
                    x3 * (y1 - y2)) / 2.0)

        A1= area(x1,y1,x2,y2,x,y)
        A2= area(x3,y3,x2,y2,x,y)
        
        A3= area(x3,y3,x4,y4,x,y)
        A4= area(x1,y1,x4,y4,x,y)
        
        A = (area(x1, y1, x2, y2, x3, y3) + area(x1, y1, x4, y4, x3, y3))
        
        return A==(A1+A2+A3+A4)

    def isPointPresentOutside(self):
        return not self.isPointPresentInside()


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
