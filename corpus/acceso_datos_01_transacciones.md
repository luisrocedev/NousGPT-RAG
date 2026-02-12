# Transacciones ACID en bases de datos

Una transacción es un conjunto de operaciones SQL que deben ejecutarse como una unidad lógica.

Propiedades ACID:
- Atomicidad: todo o nada.
- Consistencia: la BD pasa de un estado válido a otro válido.
- Aislamiento: las transacciones concurrentes no se interfieren indebidamente.
- Durabilidad: una vez confirmada, la transacción persiste.

En SQL se usan COMMIT y ROLLBACK para confirmar o deshacer cambios.
