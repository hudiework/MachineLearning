# coding:UTF-8
class Member:
    def __init__(self, **kwargs):
        self.__name = kwargs.get("name")
        self.__age = kwargs.get("age")

    def get_info(self):
        return "Member姓名 ：%s 年龄 : %s" % (self.__name, self.__age)

    def set_car(self, car):
        self.__car = car

    def get_car(self):
        return self.__car


class Car:
    def __init__(self, **kwargs):
        self.__brand = kwargs.get("brand")
        self.__price = kwargs.get("price")

    def get_info(self):
        return "CAR 品牌: %s 价格: %s" % (self.__brand, self.__price)

    def set_member(self, member):
        self.__member = member

    def get_member(self):
        return self.__member


def main():
    mem = Member(name="hudie", age=18)
    car = Car(brand="benchi",price=158)
    mem.set_car(car)
    car.set_member(mem)
    print(car.get_member().get_info())
    print(mem.get_car().get_info())


if __name__ == "__main__":
    main()
