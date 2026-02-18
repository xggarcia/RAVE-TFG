# 🎛️ Controles en Tiempo Real - RAVE Streaming

## 🎮 Descripción

El modo de **Generación en Tiempo Real** ahora incluye controles interactivos que te permiten modificar el sonido mientras se está generando, sin interrumpir el streaming.

## ⌨️ Controles de Teclado

### Volumen / Ganancia
- **Q**: Subir volumen (+5%)
- **A**: Bajar volumen (-5%)

*Rango: 0.0 (silencio) a 1.0 (volumen máximo)*

### Temperature (Variación del Sonido)
- **W**: Aumentar temperatura (+0.1)
- **S**: Disminuir temperatura (-0.1)

*Rango: 0.1 (sonido más estable/predecible) a 3.0 (sonido más caótico/variado)*

**¿Qué hace la temperatura?**
- **Baja (0.1-0.5)**: Sonidos más suaves, predecibles, similares
- **Media (0.8-1.2)**: Balance entre variedad y coherencia
- **Alta (1.5-3.0)**: Sonidos más extremos, impredecibles, experimentales

### Smoothing (Suavizado/Interpolación)
- **E**: Aumentar suavizado (+5%)
- **D**: Disminuir suavizado (-5%)

*Rango: 0.0 (sin suavizado) a 0.95 (muy suavizado)*

**¿Qué hace el smoothing?**
- **0.0**: Cada chunk es completamente nuevo (puede sonar entrecortado)
- **0.5**: Mezcla 50% del anterior con 50% nuevo (transiciones suaves)
- **0.9**: Cambios muy graduales (sonido más continuo y fluido)

### Otros Controles
- **R**: Reset - Restaurar valores por defecto
  - Gain: 0.9
  - Temperature: 1.0
  - Smoothing: 0.0

- **ESPACIO**: Mostrar parámetros actuales

- **X** o **ESC**: Salir del streaming

## 🚀 Cómo Usar

### Desde el Menú Interactivo

```bash
python main.py
# Selecciona opción C
# Cuando pregunte "Habilitar controles en tiempo real?", presiona Enter o 's'
```

### Desde la Línea de Comandos

```bash
# Con controles interactivos (por defecto)
python main.py stream

# Sin controles interactivos (modo automático)
python main.py stream --no-interactive
```

## 🎨 Ejemplos de Uso Creativo

### Paisaje Sonoro Suave
```
1. Iniciar streaming
2. Presionar 'S' varias veces (Temperature ~0.3)
3. Presionar 'E' varias veces (Smoothing ~0.8)
4. Ajustar volumen con Q/A según preferencia
```
**Resultado**: Texturas ambientales suaves y continuas

### Glitch/Experimental
```
1. Iniciar streaming
2. Presionar 'W' múltiples veces (Temperature ~2.5)
3. Mantener Smoothing bajo (D si es necesario)
4. Volumen moderado
```
**Resultado**: Sonidos caóticos, glitchy, impredecibles

### Transiciones Graduales
```
1. Iniciar con valores por defecto
2. Presionar 'E' hasta Smoothing ~0.9
3. Ir aumentando Temperature gradualmente (W)
4. Observar cómo el sonido evoluciona lentamente
```
**Resultado**: Evolución sonora lenta y orgánica

## 🔧 Parámetros Técnicos

| Parámetro | Rango | Default | Efecto |
|-----------|-------|---------|--------|
| **Gain** | 0.0 - 1.0 | 0.9 | Amplitud de salida |
| **Temperature** | 0.1 - 3.0 | 1.0 | Escala del ruido latente |
| **Smoothing** | 0.0 - 0.95 | 0.0 | Interpolación entre chunks |

### Fórmulas Aplicadas

**Temperature:**
```python
z_random = torch.randn(...) * temperature
# temperature > 1: más extremo
# temperature < 1: más contenido
```

**Smoothing:**
```python
z_current = smoothing * z_previous + (1 - smoothing) * z_new
# smoothing = 0: sin memoria
# smoothing = 0.9: 90% del anterior, 10% nuevo
```

**Gain:**
```python
audio_out = (audio_normalized) * gain
# Controla el volumen final
```

## 💡 Tips

1. **Experimenta gradualmente**: Cambia un parámetro a la vez para entender su efecto

2. **Combina parámetros**: Los efectos más interesantes surgen de combinaciones
   - Alta temperatura + alto smoothing = caos controlado
   - Baja temperatura + bajo smoothing = minimalismo digital

3. **Usa Reset (R)**: Si te pierdes, presiona R para volver a valores conocidos

4. **Graba tu sesión**: Usa software de grabación (Audacity, OBS) para capturar sesiones interesantes

5. **Performance en vivo**: Úsalo como instrumento en presentaciones o instalaciones

## 🎯 Casos de Uso

- **Composición generativa**: Explora espacios sonoros y graba fragmentos interesantes
- **Performance en vivo**: Control expresivo en tiempo real
- **Sound design**: Genera texturas únicas para producciones
- **Instalaciones**: Audio reactivo/generativo para exposiciones
- **Meditación/Ambient**: Paisajes sonoros infinitos personalizables

## 🐛 Troubleshooting

**Los controles no responden**
- Asegúrate de que la ventana del terminal esté en foco
- En Windows, los controles usan `msvcrt` (nativo)

**Cambios muy bruscos**
- Aumenta el Smoothing (tecla E)
- Reduce la Temperature (tecla S)

**Sonido muy silencioso**
- Aumenta el Gain (tecla Q)
- Verifica el volumen del sistema

**Audio entrecortado**
- Reduce la Temperature
- Aumenta el chunk_duration a 1.5 o 2.0 segundos

---

**¡Disfruta explorando el espacio latente de RAVE en tiempo real!** 🎵✨
