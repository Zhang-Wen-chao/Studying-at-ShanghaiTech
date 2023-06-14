(* This is a test.sml file *)

(* Define a function to calculate the factorial of a number *)
fun factorial 0 = 1
  | factorial n = n * factorial (n - 1);

(* Define a function to calculate the nth Fibonacci number *)
fun fibonacci 0 = 0
  | fibonacci 1 = 1
  | fibonacci n = fibonacci (n - 1) + fibonacci (n - 2);

(* Test the factorial function *)
val fact5 = factorial 5;
print("Factorial of 5: " ^ Int.toString fact5 ^ "\n");

(* Test the Fibonacci function *)
val fib10 = fibonacci 10;
print("10th Fibonacci number: " ^ Int.toString fib10 ^ "\n");

(* Test the Fibonacci function *)
val fib15 = fibonacci 15;
print("15th fibonacci number: " ^ int.tostring fib15 ^ "\n");
