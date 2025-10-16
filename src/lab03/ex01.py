def normalize(text: str, *, casefold: bool = True, yo2e: bool = True) -> str:
    if text is None:
        raise ValueError
    if not isinstance(text, str):
        raise TypeError
    if len(text) == 0:
        return "" 
    if casefold:
        text= text.casefold()
    if yo2e:
        text=text.replace('ё', 'е').replace('Ё','Е')
    text=text.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')
    while '  ' in text:
        text=text.replace('  ', ' ')
    text= text.strip()
    return text
print(normalize("ПрИвЕт\nМИр\t"))
print(normalize("ёжик, Ёлка")) 
print(normalize("Hello\r\nWorld"))
print(normalize("  двойные   пробелы  "))

