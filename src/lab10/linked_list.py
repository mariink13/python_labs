from typing import Any, Optional, Iterator

class Node:
    # Узел связного списка
    def __init__(self, value: Any):
        self.value = value  # Значение узла
        self.next = None    # Ссылка на следующий узел
    
    def __str__(self):
        return f"Node({self.value})"


class SinglyLinkedList:
    # Односвязный список
    def __init__(self):
        self.head = None  # Первый узел
        self.tail = None  # Последний узел
        self._size = 0    # Количество элементов
    
    def append(self, value: Any) -> None:
        # Добавить в конец за O(1)
        new_node = Node(value)
        
        if not self.head:  # Если список пуст
            self.head = new_node
            self.tail = new_node
        else:  # Если есть элементы
            self.tail.next = new_node
            self.tail = new_node
        
        self._size += 1
    
    def prepend(self, value: Any) -> None:
        # Добавить в начало за O(1)
        new_node = Node(value)
        
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head = new_node
        
        self._size += 1
    
    def insert(self, idx: int, value: Any) -> None:
        # Вставить по индексу
        if idx < 0 or idx > self._size:
            raise IndexError(f"Индекс {idx} вне диапазона")
        
        if idx == 0:
            self.prepend(value)
        elif idx == self._size:
            self.append(value)
        else:
            new_node = Node(value)
            current = self.head
            
            # Переходим к узлу перед нужной позицией
            for _ in range(idx - 1):
                current = current.next
            
            # Вставляем новый узел
            new_node.next = current.next
            current.next = new_node
            self._size += 1
    
    def remove(self, value: Any) -> bool:
        # Удалить по значению
        if not self.head:
            return False
        
        # Удаление из начала
        if self.head.value == value:
            self.head = self.head.next
            if not self.head:  # Если список стал пустым
                self.tail = None
            self._size -= 1
            return True
        
        # Поиск элемента для удаления
        current = self.head
        while current.next and current.next.value != value:
            current = current.next
        
        # Элемент не найден
        if not current.next:
            return False
        
        # Удаление элемента
        if current.next == self.tail:
            self.tail = current
        current.next = current.next.next
        self._size -= 1
        return True
    
    def remove_at(self, idx: int) -> Any:
        # Удалить по индексу
        if idx < 0 or idx >= self._size:
            raise IndexError(f"Индекс {idx} вне диапазона")
        
        # Удаление из начала
        if idx == 0:
            value = self.head.value
            self.head = self.head.next
            if not self.head:
                self.tail = None
            self._size -= 1
            return value
        
        # Поиск узла перед удаляемым
        current = self.head
        for _ in range(idx - 1):
            current = current.next
        
        # Удаление
        value = current.next.value
        if current.next == self.tail:
            self.tail = current
        current.next = current.next.next
        self._size -= 1
        return value
    
    def __iter__(self) -> Iterator[Any]:
        # Итератор по значениям
        current = self.head
        while current:
            yield current.value
            current = current.next
    
    def __len__(self) -> int:
        # Количество элементов
        return self._size
    
    def __str__(self) -> str:
        # Визуальное представление
        if not self.head:
            return "None"
        
        parts = []
        current = self.head
        while current:
            parts.append(f"[{current.value}]")
            current = current.next
        
        return " -> ".join(parts) + " -> None"


sll = SinglyLinkedList()
print(f'Длина нашего односвязанного списка : {len(sll)}')

sll.append(1)
sll.append(2)
sll.prepend(0)
print(f'Наша ныняшняя длина списка после добавления эллементов : {len(sll)}') 
print(f'Односвязаный список : {list(sll)}')

sll.insert(1, 0.5)
print(f'Длина списка после добавления на 1 индекс числа 0.5 : {len(sll)}')
print(f'Односвязаный список : {list(sll)}')
sll.append(52)
print(f'Односвязанный список после добавления числа в конец : {list(sll)}')

print(sll) 