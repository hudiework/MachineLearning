# coding:UTF-8
class Channel:
    def build(self):
        print("[Channel]通道连接。。。")


class DatabaseChannel(Channel):
    def build(self):
      print("[DatabaseChannel]数据库通道建立连接。。。")


def main():
    channel = DatabaseChannel()
    channel.build()


if __name__ == "__main__":
    main()