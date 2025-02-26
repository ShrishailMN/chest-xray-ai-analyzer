import sqlite3

def view_database():
    try:
        # Connect to database
        conn = sqlite3.connect('reports.db')
        cursor = conn.cursor()
        
        # Get all records
        cursor.execute('SELECT * FROM reports')
        records = cursor.fetchall()
        
        # Print records in a formatted way
        print("\n=== Stored Reports ===")
        for record in records:
            print("\n------------------------")
            print(f"ID: {record[0]}")
            print(f"Patient Name: {record[1]}")
            print(f"Patient ID: {record[2]}")
            print(f"Age: {record[3]}")
            print(f"Gender: {record[4]}")
            print(f"Exam Date: {record[5]}")
            print(f"Condition: {record[6]}")
            print(f"Confidence: {record[7]:.1%}")
            print(f"Created At: {record[11]}")
            
        print(f"\nTotal Records: {len(records)}")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    view_database() 