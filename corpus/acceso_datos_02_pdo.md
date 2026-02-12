# Acceso a MySQL con PDO

PDO permite conectarse a diferentes motores con una API uniforme.

Ejemplo básico:
- Crear DSN: mysql:host=localhost;dbname=academia;charset=utf8mb4
- Instanciar PDO con usuario y contraseña
- Configurar ERRMODE_EXCEPTION para capturar errores

Para evitar inyección SQL se deben usar sentencias preparadas con prepare y execute.
Nunca concatenar directamente entradas de usuario en una consulta SQL.
