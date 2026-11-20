"""Creates ecommerce.db: a small synthetic fashion e-commerce database (SQLite)."""
import os
import random
import sqlite3
from datetime import date, timedelta

random.seed(42)
DB = os.getenv("DB_PATH", "ecommerce.db")
if os.path.exists(DB):
    os.remove(DB)

con = sqlite3.connect(DB)
con.executescript("""
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT, state TEXT,
    signup_date TEXT, segment TEXT
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL, cost REAL
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id),
    order_date TEXT, status TEXT, channel TEXT
);
CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY, order_id INTEGER REFERENCES orders(order_id),
    product_id INTEGER REFERENCES products(product_id), quantity INTEGER, discount_pct REAL
);
""")

CITIES = [("Delhi", "Delhi"), ("Noida", "Uttar Pradesh"), ("Gurugram", "Haryana"),
          ("Mumbai", "Maharashtra"), ("Pune", "Maharashtra"), ("Bengaluru", "Karnataka"),
          ("Hyderabad", "Telangana"), ("Chennai", "Tamil Nadu"), ("Kolkata", "West Bengal"),
          ("Jaipur", "Rajasthan")]
FIRST = ["Aarav", "Riya", "Kabir", "Ananya", "Vihaan", "Isha", "Rohan", "Meera", "Arjun",
         "Sana", "Dev", "Pooja", "Karan", "Neha", "Ishaan", "Tara", "Yash", "Kiara"]
LAST = ["Sharma", "Verma", "Gupta", "Singh", "Patel", "Iyer", "Reddy", "Khan", "Das", "Mehta"]
CATEGORIES = {  # category: (item names, price range)
    "Kurtas": (["Cotton Kurta", "Printed Kurta", "Silk Kurta"], (700, 2500)),
    "Dresses": (["Floral Dress", "Maxi Dress", "Wrap Dress"], (1200, 4500)),
    "Tops": (["Crop Top", "Linen Shirt", "Graphic Tee"], (400, 1800)),
    "Jeans": (["Slim Jeans", "Straight Jeans", "Wide-leg Jeans"], (1000, 3200)),
    "Footwear": (["Flats", "Sneakers", "Block Heels"], (800, 3500)),
    "Accessories": (["Tote Bag", "Scarf", "Belt"], (300, 2000)),
}
END = date(2026, 10, 31)

for i in range(1, 451):
    city, state = random.choice(CITIES)
    signup = date(2024, 6, 1) + timedelta(days=random.randint(0, (END - date(2024, 6, 1)).days))
    con.execute("INSERT INTO customers VALUES (?,?,?,?,?,?)",
                (i, f"{random.choice(FIRST)} {random.choice(LAST)}", city, state,
                 signup.isoformat(), random.choices(["Regular", "VIP"], [0.85, 0.15])[0]))

pid = 1
for cat, (names, (lo, hi)) in CATEGORIES.items():
    for n in range(10):
        price = round(random.randint(lo, hi) / 10) * 10
        con.execute("INSERT INTO products VALUES (?,?,?,?,?)",
                    (pid, f"{random.choice(names)} #{n + 1}", cat, price,
                     round(price * random.uniform(0.35, 0.6), 2)))
        pid += 1

signups = dict(con.execute("SELECT customer_id, signup_date FROM customers"))
item_id = 1
for oid in range(1, 3001):
    cid = random.randint(1, 420)  # last 30 customers never order
    start = date.fromisoformat(signups[cid])
    odate = start + timedelta(days=random.randint(0, (END - start).days))
    status = random.choices(["delivered", "cancelled", "returned"], [0.85, 0.08, 0.07])[0]
    channel = random.choices(["Website", "Myntra", "Amazon", "Flipkart"], [0.35, 0.25, 0.2, 0.2])[0]
    con.execute("INSERT INTO orders VALUES (?,?,?,?,?)", (oid, cid, odate.isoformat(), status, channel))
    for p in random.sample(range(1, 61), random.randint(1, 4)):
        con.execute("INSERT INTO order_items VALUES (?,?,?,?,?)",
                    (item_id, oid, p, random.randint(1, 3),
                     random.choices([0, 10, 20, 30, 40, 50], [30, 20, 20, 15, 10, 5])[0]))
        item_id += 1

con.commit()
for t in ["customers", "products", "orders", "order_items"]:
    print(t, con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0], "rows")
con.close()
