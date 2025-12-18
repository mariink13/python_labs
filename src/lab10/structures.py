from collections import deque
from typing import Any, Optional

class Stack:
    # LIFO структура - последний вошел, первый вышел
    def __init__(self, iterable=None):
        self._data = list(iterable) if iterable else []
    
    def push(self, item: Any) -> None:
        # Добавить на вершину стека
        self._data.append(item)
    
    def pop(self) -> Any:
        # Снять верхний элемент
        if not self._data:
            raise IndexError("Стек пуст")
        return self._data.pop()
    
    def peek(self) -> Optional[Any]:
        # Посмотреть верхний элемент без удаления
        return self._data[-1] if self._data else None
    
    def is_empty(self) -> bool:
        # Проверка на пустоту
        return not self._data
    
    def __len__(self) -> int:
        # Количество элементов
        return len(self._data)
    
    def __str__(self) -> str:
        return f"Stack({self._data})"


class Queue:
    # FIFO структура - первый вошел, первый вышел
    def __init__(self, iterable=None):
        self._data = deque(iterable) if iterable else deque()
    
    def enqueue(self, item: Any) -> None:
        # Добавить в конец очереди
        self._data.append(item)
    
    def dequeue(self) -> Any:
        # Взять из начала очереди
        if not self._data:
            raise IndexError("Очередь пуста")
        return self._data.popleft()
    
    def peek(self) -> Optional[Any]:
        # Посмотреть первый элемент
        return self._data[0] if self._data else None
    
    def is_empty(self) -> bool:
        # Проверка на пустоту
        return not self._data
    
    def __len__(self) -> int:
        # Количество элементов
        return len(self._data)
    
    def __str__(self) -> str:
        return f"Queue({list(self._data)})"



print('Stack')

stack = Stack([1,2,3,4])
print(f'Снятие верхнего элемента стека : {stack.pop()}')
print(f'Пустой ли стек? {stack.is_empty()}')
print(f'Число сверху : {stack.peek()}')
stack.push(1)
print(f'Значение сверху после добавления числа в стек : {stack.peek()}')
print(f'Длина стека : {len(stack)}')
print(f'Стек : {stack._data}')

print('Deque')

q = Queue([1,2,3,4])

print(f'Значение первого эллемента : {q.peek()}')
q.dequeue()
print(f'Значение первого эллемента после удаления числа : {q.peek()}')
q.enqueue(52)
print(f'Значение первого эллемента после добавления числа : {q.peek()}')
print(f'Пустая ли очередь? {q.is_empty()}')
print(f'Количество элементов в очереди : {len(q)}')