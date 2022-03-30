"""
Classes of all the physics objects
Author:
Nilusink
"""
from ursina.shaders import lit_with_shadows_shader
from ursina import *
import typing as tp
import numpy as np

# gravitational constant
G: float = 6.67408e-11

# pi
PI = 3.1415926535897932384626433832795028841971

# one astronomical unit
AU = 149597870700


class Vector:
    x: float
    y: float
    angle: float
    length: float

    # creation of new elements
    def __init__(self) -> None:
        self.__x: float = 0
        self.__y: float = 0
        self.__angle: float = 0
        self.__length: float = 0

    @staticmethod
    def from_cartesian(x: float, y: float) -> "Vector":
        p = Vector()
        p.x = x
        p.y = y

        return p

    @staticmethod
    def from_polar(angle: float, length: float) -> "Vector":
        p = Vector()
        while angle > 2*PI:
            angle -= 2*PI
        while angle < 0:
            angle += 2*PI
        p.angle = angle
        p.length = length

        return p

    # variable getters / setters
    @property
    def x(self) -> float:
        return self.__x

    @x.setter
    def x(self, value: float) -> None:
        self.__x = value
        self.__update("c")

    @property
    def y(self) -> float:
        return self.__y

    @y.setter
    def y(self, value: float) -> None:
        self.__y = value
        self.__update("c")

    @property
    def angle(self) -> float:
        """
        value in radian
        """
        return self.__angle

    @angle.setter
    def angle(self, value: float) -> None:
        """
        value in radian
        """
        self.__angle = value
        self.__update("p")

    @property
    def length(self) -> float:
        return self.__length

    @length.setter
    def length(self, value: float) -> None:
        self.__length = value
        self.__update("p")

    # maths
    def __add__(self, other) -> "Vector":
        if type(other) == Vector:
            return Vector.from_cartesian(x=self.x + other.x, y=self.y + other.y)

        return Vector.from_cartesian(x=self.x + other, y=self.y + other)

    def __sub__(self, other) -> "Vector":
        if type(other) == Vector:
            return Vector.from_cartesian(x=self.x - other.x, y=self.y - other.y)

        return Vector.from_cartesian(x=self.x - other, y=self.y - other)

    def __mul__(self, other) -> "Vector":
        if type(other) == Vector:
            return Vector.from_polar(angle=self.angle + other.angle, length=self.length * other.length)

        return Vector.from_cartesian(x=self.x * other, y=self.y * other)

    def __truediv__(self, other) -> "Vector":
        return Vector.from_cartesian(x=self.x / other, y=self.y / other)

    # internal functions
    def __update(self, calc_from: str) -> None:
        """
        :param calc_from: polar (p) | cartesian (c)
        """
        if calc_from in ("p", "polar"):
            self.__x = np.cos(self.angle) * self.length
            self.__y = np.sin(self.angle) * self.length

        elif calc_from in ("c", "cartesian"):
            self.__length = np.sqrt(self.x**2 + self.y**2)
            self.__angle = np.arctan2(self.y, self.x)
            return

        else:
            raise ValueError("Invalid value for \"calc_from\"")

    def __abs__(self) -> float:
        return np.sqrt(self.x**2 + self.y**2)

    def __repr__(self) -> str:
        return f"<Vector: x={self.x}, y={self.y}>"


class Vector3D:
    """
    Simple 3D vector class
    """
    x: float
    y: float
    z: float
    angle_xy: float
    angle_xz: float
    length_xy: float
    length: float

    def __init__(self):
        self.__x: float = 0
        self.__y: float = 0
        self.__z: float = 0
        self.__angle_xy: float = 0
        self.__angle_xz: float = 0
        self.__length_xy: float = 0
        self.__length: float = 0

    @property
    def x(self) -> float:
        return self.__x

    @x.setter
    def x(self, value: float) -> None:
        self.__x = value
        self.__update("c")

    @property
    def y(self) -> float:
        return self.__y

    @y.setter
    def y(self, value: float) -> None:
        self.__y = value
        self.__update("c")

    @property
    def z(self) -> float:
        return self.__z

    @z.setter
    def z(self, value: float) -> None:
        self.__z = value
        self.__update("c")

    @property
    def cartesian(self) -> tp.Tuple[float, float, float]:
        """
        :return: x, y, z
        """
        return self.x, self.y, self.z

    @cartesian.setter
    def cartesian(self, value: tp.Tuple[float, float, float]) -> None:
        """
        :param value: (x, y, z)
        """
        self.__x, self.__y, self.__z = value
        self.__update("c")

    @property
    def angle_xy(self) -> float:
        return self.__angle_xy

    @angle_xy.setter
    def angle_xy(self, value: float) -> None:
        self.__angle_xy = self.normalize_angle(value)
        self.__update("p")

    @property
    def angle_xz(self) -> float:
        return self.__angle_xz

    @angle_xz.setter
    def angle_xz(self, value: float) -> None:
        self.__angle_xz = self.normalize_angle(value)
        self.__update("p")

    @property
    def length_xy(self) -> float:
        """
        can't be set
        """
        return self.__length_xy

    @property
    def length(self) -> float:
        return self.__length

    @length.setter
    def length(self, value: float) -> None:
        self.__length = value
        self.__update("p")

    @property
    def polar(self) -> tp.Tuple[float, float, float]:
        """
        :return: angle_xy, angle_xz, length
        """
        return self.angle_xy, self.angle_xz, self.length

    @polar.setter
    def polar(self, value: tp.Tuple[float, float, float]) -> None:
        """
        :param value: (angle_xy, angle_xz, length)
        """
        self.__angle_xy = self.normalize_angle(value[0])
        self.__angle_xz = self.normalize_angle(value[1])
        self.__length = value[2]
        self.__update("p")

    @staticmethod
    def from_polar(angle_xy: float, angle_xz: float, length: float) -> "Vector3D":
        """
        create a Vector3D from polar form
        """
        v = Vector3D()
        v.polar = angle_xy, angle_xz, length
        return v

    @staticmethod
    def from_cartesian(x: float, y: float, z: float) -> "Vector3D":
        """
        create a Vector3D from cartesian form
        """
        v = Vector3D()
        v.cartesian = x, y, z
        return v

    @staticmethod
    def calculate_with_angles(length: float, angle1: float, angle2: float) -> tp.Tuple[float, float, float]:
        """
        calculate the x, y and z components of length facing (angle1, angle2)
        """
        tmp = np.cos(angle2) * length
        z = np.sin(angle2) * length
        x = np.cos(angle1) * tmp
        y = np.sin(angle1) * tmp

        return x, y, z

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        removes "overflow" from an angle
        """
        while angle > 2 * PI:
            angle -= 2 * PI

        while angle < 0:
            angle += 2 * PI

        return angle

    # maths
    def __neg__(self) -> "Vector3D":
        self.cartesian = [-el for el in self.cartesian]
        return self

    def __add__(self, other) -> "Vector3D":
        if type(other) == Vector3D:
            return Vector3D.from_cartesian(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

        return Vector3D.from_cartesian(x=self.x + other, y=self.y + other, z=self.z + other)

    def __sub__(self, other) -> "Vector3D":
        if type(other) == Vector3D:
            return Vector3D.from_cartesian(x=self.x - other.x, y=self.y - other.y, z=self.z - other.z)

        return Vector3D.from_cartesian(x=self.x - other, y=self.y - other, z=self.z - other)

    def __mul__(self, other) -> "Vector3D":
        if type(other) == Vector3D:
            return Vector3D.from_polar(angle_xy=self.angle_xy + other.angle_xy, angle_xz=self.angle_xz + other.angle_xz, length=self.length * other.length)

        return Vector3D.from_cartesian(x=self.x * other, y=self.y * other, z=self.z * other)

    def __truediv__(self, other) -> "Vector3D":
        return Vector3D.from_cartesian(x=self.x / other, y=self.y / other, z=self.z / other)

    # internal functions
    def __update(self, calc_from: str) -> None:
        match calc_from:
            case "p":
                self.__length_xy = np.cos(self.angle_xz) * self.length
                x, y, z = self.calculate_with_angles(self.length, self.angle_xy, self.angle_xz)
                self.__x = x
                self.__y = y
                self.__z = z

            case "c":
                self.__length_xy = np.sqrt(self.y**2 + self.x**2)
                self.__angle_xy = np.arctan2(self.y, self.x)
                self.__angle_xz = np.arctan2(self.z, self.x)
                self.__length = np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __repr__(self) -> str:
        return f"<\n" \
               f"\tVector3D:\n" \
               f"\tx:{self.x}\ty:{self.y}\tz:{self.z}\n" \
               f"\tangle_xy:{self.angle_xy}\tangle_xz:{self.__angle_xz}\tlength:{self.length}\n" \
               f">"


class BasicObject:
    def __init__(self, mass: float,
                 position: Vector3D,
                 velocity: Vector3D = Vector3D.from_cartesian(0, 0, 0),
                 acceleration: Vector3D = Vector3D.from_cartesian(0, 0, 0),
                 fixed: bool = False) -> None:
        """
        Basic physics object
        :param mass: mass
        :param position: x start position
        :param velocity: start velocity
        :param acceleration: the start acceleration of the object
        :param fixed: if true, the object won't be moved in the simulation
        """
        self.__mass = mass
        self.__position = position
        self.__trace: tp.List[Vector3D] = []
        self.velocity = velocity
        self.acceleration = acceleration
        self.fixed = fixed

    @property
    def mass(self) -> float:
        return self.__mass

    @property
    def position(self) -> Vector3D:
        return self.__position

    @position.setter
    def position(self, pos: Vector3D) -> None:
        self.__trace.append(pos)
        self.__position = pos

    @property
    def trace(self) -> tp.List[Vector3D]:
        return self.__trace


class Planet(BasicObject):
    def __init__(self, name: str, diameter: float, *args, **kw):
        super().__init__(*args, **kw)
        self.__name = name
        self.__d = diameter

    @property
    def name(self) -> str:
        return self.__name

    @property
    def diameter(self) -> float:
        return self.__d


class Simulation:
    def __init__(self, objects: tp.List[Planet] | tp.Tuple[Planet]) -> None:
        """
        All Objects to simulate should be in this class
        """
        self.__objects = objects
        self.__last_collided = [
            [],
            [],
            []
        ]

        # create objects for rendering
        self.__urs_objects = []
        for obj in self.__objects:
            self.__urs_objects.append(
                Entity(model="sphere",
                       color=color.gray,
                       x=obj.position.x,
                       z=obj.position.z,
                       y=-obj.position.y,
                       origin=(0, 0),
                       scale=[obj.diameter]*3,
                       shader=lit_with_shadows_shader
                       )
            )

    @property
    def objects(self) -> list:
        return self.__objects

    @property
    def total_mass(self) -> float:
        return sum([obj.mass for obj in self.objects])

    @property
    def max_mass(self) -> float:
        return max([obj.mass for obj in self.objects])

    @property
    def size(self) -> Vector:
        """
        The total size in x and y
        """
        x_vals = [obj.position.x for obj in self.objects]
        y_vals = [obj.position.y for obj in self.objects]

        return Vector.from_cartesian(x=max(x_vals)-min(x_vals), y=max(y_vals)-min(y_vals))

    @property
    def gravity_center(self) -> Vector:
        gx = sum([obj.position.x*obj.mass for obj in self.objects])
        gx /= self.total_mass
        gy = sum([obj.position.y*obj.mass for obj in self.objects])
        gy /= self.total_mass

        return Vector.from_cartesian(gx, gy)

    def iter(self, dt: float, gravity: bool = True, collision: bool = True, precision: int = 2) -> None:
        """
        run 1 iteration of the simulation
        """
        # iterate each object and then calculate all the forces to the other objects
        # based on F = G*(m1*m2)/r**2
        dt /= precision
        for _ in range(precision):
            if gravity:
                done_objects = []
                for now_object in self.objects.copy():
                    for influence_object in self.objects:
                        if influence_object is not now_object and not {now_object, influence_object} in done_objects:
                            done_objects.append({now_object, influence_object})

                            f_l = G * (now_object.mass * influence_object.mass)
                            delta = now_object.position - influence_object.position
                            f_l = f_l / delta.length ** 2

                            now_a = Vector3D.from_polar(
                                angle_xy=delta.angle_xy,
                                angle_xz=delta.angle_xz,
                                length=f_l / now_object.mass
                            )
                            inf_a = -Vector3D.from_polar(
                                angle_xy=delta.angle_xy,
                                angle_xz=delta.angle_xz,
                                length=f_l / influence_object.mass
                            )

                            now_object.acceleration = now_a
                            now_object.velocity += now_object.acceleration * dt

                            influence_object.acceleration = inf_a
                            influence_object.velocity += influence_object.acceleration * dt

            if collision:
                done_objects = []
                for now_object in self.objects:
                    if type(now_object) == Planet:
                        if now_object.position.z-now_object.diameter/2 <= -1:
                            now_object.acceleration = 0
                            now_object.velocity.angle_xz *= -1
                            continue

                        for influence_object in self.objects:
                            if type(influence_object) == Planet and influence_object is not now_object:
                                now_object: Planet
                                influence_object: Planet
                                delta = now_object.position - influence_object.position

                                # check if they touch
                                if delta.length < now_object.diameter/2 + influence_object.diameter/2 and not now_object in done_objects\
                                        and not any([{now_object, influence_object} in self.__last_collided[i] for i in range(len(self.__last_collided))]):
                                    # Formula for 1-dimensional collision: v1' = (m1*v1 + m2*(2*v2-v1)) / (m1+m2)
                                    done_objects += [now_object, influence_object]
                                    self.__last_collided.append({now_object, influence_object})

                                    # split the velocities in three directions (90°)
                                    # now object
                                    a = delta.angle_xy - now_object.velocity.angle_xy
                                    b = delta.angle_xz - now_object.velocity.angle_xz
                                    # seperate the vector in three different directions (two to be ignored, one to be calculated with)
                                    x, y, z = Vector3D.calculate_with_angles(now_object.velocity.length, a, b)

                                    now_collision = Vector3D.from_polar(angle_xy=delta.angle_xy,
                                                                        angle_xz=delta.angle_xz,
                                                                        length=x)

                                    now_carry1 = Vector3D.from_polar(angle_xy=delta.angle_xy-PI/2,
                                                                     angle_xz=delta.angle_xz,
                                                                     length=y)

                                    now_carry2 = Vector3D.from_polar(angle_xy=delta.angle_xy,
                                                                     angle_xz=delta.angle_xz-PI/2,
                                                                     length=z)

                                    # collision object
                                    a = delta.angle_xy - influence_object.velocity.angle_xy
                                    b = delta.angle_xz - influence_object.velocity.angle_xz
                                    x, y, z = Vector3D.calculate_with_angles(influence_object.velocity.length, a, b)

                                    inf_collision = -Vector3D.from_polar(angle_xy=delta.angle_xy,
                                                                        angle_xz=delta.angle_xz,
                                                                        length=x)

                                    inf_carry1 = -Vector3D.from_polar(angle_xy=delta.angle_xy-PI/2,
                                                                     angle_xz=delta.angle_xz,
                                                                     length=y)

                                    inf_carry2 = -Vector3D.from_polar(angle_xy=delta.angle_xy,
                                                                     angle_xz=delta.angle_xz-PI/2,
                                                                     length=z)

                                    # 1 dimensional collision calculation
                                    now_v = now_collision.length * now_object.mass
                                    now_v += (inf_collision.length * 2 - now_collision.length) * influence_object.mass
                                    now_v /= now_object.mass + influence_object.mass

                                    inf_v = inf_collision.length * influence_object.mass
                                    inf_v += (now_collision.length * 2 - inf_collision.length) * now_object.mass
                                    inf_v /= influence_object.mass + now_object.mass

                                    now_v = now_carry1 + now_carry2 + Vector3D.from_polar(angle_xy=now_collision.angle_xy, angle_xz=now_collision.angle_xz, length=now_v)
                                    inf_v = inf_carry1 + inf_carry2 + Vector3D.from_polar(angle_xy=inf_collision.angle_xy, angle_xz=inf_collision.angle_xz, length=inf_v)

                                    # assign velocities to objects
                                    now_object.acceleration = Vector3D()
                                    now_object.velocity = now_v

                                    influence_object.acceleration = Vector3D()
                                    influence_object.velocity = inf_v

                # move 1 down
                self.__last_collided = [
                    [],
                    [self.__last_collided[0]],
                    [self.__last_collided[1]]
                ]

            for now_object, ursina_obj in zip(self.objects, self.__urs_objects):
                if not now_object.fixed:
                    now_object.velocity += now_object.acceleration * dt
                    now_object.position += now_object.velocity * dt

                    ursina_obj.x = now_object.position.x
                    ursina_obj.y = now_object.position.z
                    ursina_obj.z = now_object.position.y


if __name__ == "__main__":
    v = Vector3D.from_cartesian(2, 2, 2)
    print(v)