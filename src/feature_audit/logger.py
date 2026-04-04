class AuditLogger:
    @staticmethod
    def step(title: str) -> None:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    @staticmethod
    def info(message: str) -> None:
        print(message)

    @staticmethod
    def kv(key: str, value) -> None:
        print(f"{key}: {value}")
