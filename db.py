import sqlite3

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# cursor.execute("CREATE TABLE drinks(drink, price)")

many_drinks = [
    ('black tea', 230),
    ('coffee', 240),
    ('milk', 430),
    ('ice coffee', 450),
]
cursor.execute("INSERT INTO drinks VALUES (?,?)", many_drinks)

conn.commit()

conn.close()