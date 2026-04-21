section .data

section .text

extern print

global _exit


_exit:
  mov rax, 60
  mov rdi, 0
  syscall

section .bss