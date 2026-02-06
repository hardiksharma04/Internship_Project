import numpy as np
import sys

class MatrixOperationsTool:
    """A comprehensive tool for matrix operations using NumPy"""
    
    def __init__(self):
        self.matrices = {}
        self.history = []
        
    def display_header(self, title):
        """Display formatted header"""
        print("\n" + "="*60)
        print(f"  {title.center(56)}")
        print("="*60)
    
    def display_matrix(self, matrix, name="Matrix"):
        """Display matrix in structured format"""
        print(f"\n{name}:")
        print("-" * 40)
        with np.printoptions(precision=4, suppress=True, threshold=100):
            print(matrix)
        print(f"Shape: {matrix.shape}")
    
    def input_matrix(self, rows, cols, name="Matrix"):
        """Input matrix elements with validation"""
        print(f"\nEnter elements for {name} ({rows}x{cols}):")
        print(f"(Enter {cols} space-separated numbers per row)")
        matrix = []
        
        for i in range(rows):
            while True:
                try:
                    row_input = input(f"Row {i+1}: ").strip()
                    if not row_input:
                        print("Row cannot be empty. Try again.")
                        continue
                    row = list(map(float, row_input.split()))
                    if len(row) != cols:
                        print(f"Expected {cols} elements, got {len(row)}. Try again.")
                        continue
                    matrix.append(row)
                    break
                except ValueError:
                    print("Invalid input. Please enter numbers only.")
        
        return np.array(matrix)
    
    def add_matrices(self, name_a, name_b):
        """Add two matrices"""
        if name_a not in self.matrices or name_b not in self.matrices:
            print("Error: One or both matrices not found.")
            return
        
        A = self.matrices[name_a]
        B = self.matrices[name_b]
        
        if A.shape != B.shape:
            print(f"Error: Matrix shapes don't match. {A.shape} vs {B.shape}")
            return
        
        result = A + B
        self.display_header(f"Addition: {name_a} + {name_b}")
        self.display_matrix(A, name_a)
        print()
        self.display_matrix(B, name_b)
        print()
        self.display_matrix(result, "Result (A + B)")
        self.history.append(f"Addition: {name_a} + {name_b}")
    
    def subtract_matrices(self, name_a, name_b):
        """Subtract two matrices"""
        if name_a not in self.matrices or name_b not in self.matrices:
            print("Error: One or both matrices not found.")
            return
        
        A = self.matrices[name_a]
        B = self.matrices[name_b]
        
        if A.shape != B.shape:
            print(f"Error: Matrix shapes don't match. {A.shape} vs {B.shape}")
            return
        
        result = A - B
        self.display_header(f"Subtraction: {name_a} - {name_b}")
        self.display_matrix(A, name_a)
        print()
        self.display_matrix(B, name_b)
        print()
        self.display_matrix(result, "Result (A - B)")
        self.history.append(f"Subtraction: {name_a} - {name_b}")
    
    def multiply_matrices(self, name_a, name_b):
        """Multiply two matrices (dot product)"""
        if name_a not in self.matrices or name_b not in self.matrices:
            print("Error: One or both matrices not found.")
            return
        
        A = self.matrices[name_a]
        B = self.matrices[name_b]
        
        if A.shape[1] != B.shape[0]:
            print(f"Error: Cannot multiply. {A.shape[1]} != {B.shape[0]}")
            print(f"Matrix A columns must equal Matrix B rows.")
            return
        
        result = np.dot(A, B)
        self.display_header(f"Multiplication: {name_a} × {name_b}")
        self.display_matrix(A, name_a)
        print()
        self.display_matrix(B, name_b)
        print()
        self.display_matrix(result, "Result (A × B)")
        self.history.append(f"Multiplication: {name_a} × {name_b}")
    
    def transpose_matrix(self, name):
        """Calculate transpose of a matrix"""
        if name not in self.matrices:
            print("Error: Matrix not found.")
            return
        
        matrix = self.matrices[name]
        result = matrix.T
        
        self.display_header(f"Transpose of {name}")
        self.display_matrix(matrix, name)
        print()
        self.display_matrix(result, f"Transpose of {name}")
        self.history.append(f"Transpose: {name}")
    
    def determinant_matrix(self, name):
        """Calculate determinant of a square matrix"""
        if name not in self.matrices:
            print("Error: Matrix not found.")
            return
        
        matrix = self.matrices[name]
        
        if matrix.shape[0] != matrix.shape[1]:
            print(f"Error: Determinant is only defined for square matrices.")
            print(f"Current matrix shape: {matrix.shape}")
            return
        
        try:
            det = np.linalg.det(matrix)
            self.display_header(f"Determinant of {name}")
            self.display_matrix(matrix, name)
            print(f"\n{'Determinant:'.ljust(20)} {det:.6f}")
            self.history.append(f"Determinant: {name}")
        except np.linalg.LinAlgError:
            print("Error: Matrix is singular (determinant cannot be calculated).")
    
    def list_matrices(self):
        """List all stored matrices"""
        if not self.matrices:
            print("No matrices stored yet.")
            return
        
        self.display_header("Stored Matrices")
        for name, matrix in self.matrices.items():
            print(f"  • {name}: {matrix.shape}")
    
    def view_matrix(self, name):
        """View a specific matrix"""
        if name not in self.matrices:
            print(f"Matrix '{name}' not found.")
            return
        
        self.display_header(f"Matrix: {name}")
        self.display_matrix(self.matrices[name], name)
    
    def main_menu(self):
        """Display main menu"""
        self.display_header("MATRIX OPERATIONS TOOL")
        print("\nMain Menu:")
        print("  1. Input New Matrix")
        print("  2. List All Matrices")
        print("  3. View Matrix")
        print("  4. Add Two Matrices")
        print("  5. Subtract Two Matrices")
        print("  6. Multiply Two Matrices")
        print("  7. Transpose Matrix")
        print("  8. Calculate Determinant")
        print("  9. View Operation History")
        print("  0. Exit")
        print("-" * 60)
    
    def run(self):
        """Run the application"""
        self.display_header("Welcome to Matrix Operations Tool")
        print("\nThis tool allows you to perform various matrix operations")
        print("including addition, subtraction, multiplication, transpose,")
        print("and determinant calculation using NumPy.\n")
        
        while True:
            try:
                self.main_menu()
                choice = input("Enter your choice (0-9): ").strip()
                
                if choice == "1":
                    name = input("Enter matrix name (e.g., A, B, C): ").strip().upper()
                    if not name:
                        print("Matrix name cannot be empty.")
                        continue
                    
                    rows = int(input("Enter number of rows: "))
                    cols = int(input("Enter number of columns: "))
                    
                    if rows <= 0 or cols <= 0:
                        print("Rows and columns must be positive integers.")
                        continue
                    
                    self.matrices[name] = self.input_matrix(rows, cols, name)
                    print(f"Matrix {name} stored successfully!")
                
                elif choice == "2":
                    self.list_matrices()
                
                elif choice == "3":
                    name = input("Enter matrix name to view: ").strip().upper()
                    self.view_matrix(name)
                
                elif choice == "4":
                    name_a = input("Enter first matrix name: ").strip().upper()
                    name_b = input("Enter second matrix name: ").strip().upper()
                    self.add_matrices(name_a, name_b)
                
                elif choice == "5":
                    name_a = input("Enter first matrix name: ").strip().upper()
                    name_b = input("Enter second matrix name: ").strip().upper()
                    self.subtract_matrices(name_a, name_b)
                
                elif choice == "6":
                    name_a = input("Enter first matrix name: ").strip().upper()
                    name_b = input("Enter second matrix name: ").strip().upper()
                    self.multiply_matrices(name_a, name_b)
                
                elif choice == "7":
                    name = input("Enter matrix name: ").strip().upper()
                    self.transpose_matrix(name)
                
                elif choice == "8":
                    name = input("Enter matrix name: ").strip().upper()
                    self.determinant_matrix(name)
                
                elif choice == "9":
                    if not self.history:
                        print("No operations performed yet.")
                    else:
                        self.display_header("Operation History")
                        for i, operation in enumerate(self.history, 1):
                            print(f"  {i}. {operation}")
                
                elif choice == "0":
                    self.display_header("Thank You!")
                    print("Program terminated successfully. Goodbye!")
                    break
                
                else:
                    print("Invalid choice. Please enter a number between 0 and 9.")
                
            except ValueError as e:
                print(f"Invalid input. Please enter valid values.")
            except KeyboardInterrupt:
                print("\n\nProgram interrupted by user.")
                break
            except Exception as e:
                print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    tool = MatrixOperationsTool()
    tool.run()
