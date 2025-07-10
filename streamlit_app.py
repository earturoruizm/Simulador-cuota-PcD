# ==============================================================================
# CÓDIGO DEL SIMULADOR v16 (VERSIÓN CON DISEÑO VISUAL MEJORADO)
# ==============================================================================
# Código original de: Edwin Arturo Ruiz Moreno - Comisionado Nacional del Servicio Civil
# Derechos Reservados CNSC © 2025
# Adaptación, corrección y rediseño visual por: Asistente de IA de Google
# Fecha de rediseño: 2025-07-10
# ==============================================================================

# --- IMPORTACIONES ---
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import io
from datetime import datetime
from enum import Enum
from typing import Tuple, Optional, Any, Dict, List
from dataclasses import dataclass
import streamlit as st

# --- CONFIGURACIÓN E IMPORTACIONES ADICIONALES ---
# Intenta importar pytz para manejar zonas horarias de forma robusta.
try:
    from pytz import timezone
    BOGOTA_TZ = timezone('America/Bogota')
except ImportError:
    BOGOTA_TZ = None

# Intenta importar BeautifulSoup para un parseo de HTML más fiable en el PDF.
BS4_AVAILABLE = False
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    pass

# Importación para la generación de PDF.
from fpdf import FPDF, FPDFException
from fpdf.enums import XPos, YPos


# --- NOTA DE DISEÑO: PALETA DE COLORES Y ESTILO VISUAL ---
# Se ha rediseñado la paleta de colores para mejorar la estética, el contraste y la legibilidad.
# - Fondo Principal (fondo_app): Un gris muy claro para reducir el cansancio visual.
# - Contenedores (fondo_contenedor): Blanco puro para que el contenido principal resalte.
# - Colores Primarios (primario, secundario): Una gama de azules/turquesas que denotan profesionalismo.
# - Acento (acento): Un color ámbar vibrante para botones de acción y elementos clave.
# - Texto (texto_principal, texto_secundario): Tonos de gris oscuro en lugar de negro puro para una lectura más suave.
# - Gráficos (grafico_general, grafico_reserva): Colores complementarios que son visualmente distintos y agradables.
# - Estados (exito, error, advertencia): Colores semánticos para comunicar estados de forma intuitiva.
PALETA_COLORES = {
    'fondo_app': '#F0F2F6',
    'fondo_contenedor': '#FFFFFF',
    'borde': '#DEE2E6',
    'primario': '#005F73',      # Azul oscuro/petróleo
    'secundario': '#0A9396',    # Turquesa
    'acento': '#EE9B00',        # Ámbar/Naranja
    'texto_principal': '#212529',
    'texto_secundario': '#495057',
    'texto_claro': '#F8F9FA',
    'grafico_general': '#94D2BD', # Menta pálido
    'grafico_reserva': '#005F73', # Coincide con el primario para consistencia
    'exito': '#2A9D8F',
    'error': '#E76F51',
    'advertencia': '#F4A261',
}

# --- CONSTANTES GLOBALES ---
mpl.rcParams['figure.dpi'] = 150 # Mejora la resolución de los gráficos generados.
pd.set_option('display.float_format', lambda x: f'{x:.1f}')
CREDITOS_SIMULADOR = (
    "Código original de: Edwin Arturo Ruiz Moreno - Comisionado Nacional del Servicio Civil\n"
    "Derechos Reservados CNSC © 2025"
)

# ==============================================================================
# CLASES DE LÓGICA DE NEGOCIO (Sin cambios funcionales)
# ==============================================================================
class EstadoCalculo(Enum):
    NORMAL, CERO_VACANTES, AJUSTE_V1, AJUSTE_SIN_PCD = range(4)

@dataclass
class ModalidadResultados:
    total: int
    reserva: int
    general: int
    estado: EstadoCalculo

@dataclass
class ResultadosSimulacion:
    ingreso: ModalidadResultados
    ascenso: ModalidadResultados

@dataclass
class DatosEntrada:
    total_opec: int
    opcion_calculo_str: str
    hay_pcd_para_ascenso: bool
    v_ingreso: int
    v_ascenso: int

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convierte un color hexadecimal a una tupla RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# ==============================================================================
# CLASE PARA GENERACIÓN DE PDF (Con ajustes de estilo)
# ==============================================================================
class PDF_Reporte(FPDF):
    """Clase personalizada para generar el reporte en PDF con cabecera y pie de página."""
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}}', align='R')
        self.set_y(-20)
        self.set_x(self.l_margin)
        self.set_font('Helvetica', 'I', 7)
        for line in CREDITOS_SIMULADOR.replace("©", "(c)").split('\n'):
            self.cell(0, 4, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.set_text_color(0, 0, 0)

    def chapter_title(self, text: str):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(*hex_to_rgb(PALETA_COLORES['primario']))
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.set_draw_color(*hex_to_rgb(PALETA_COLORES['acento']))
        self.line(self.get_x(), self.get_y(), self.get_x() + 50, self.get_y())
        self.ln(4)
        self.set_text_color(*hex_to_rgb(PALETA_COLORES['texto_principal']))

    def chapter_body_html(self, html_content: str):
        self.set_font('Helvetica', '', 10)
        # Usa BeautifulSoup si está disponible para una mejor extracción de texto.
        if BS4_AVAILABLE:
            text = BeautifulSoup(html_content.replace("</li>", "\n"), "html.parser").get_text(separator=" ")
        else:
            # Fallback a regex si BS4 no está disponible.
            import re
            text = re.sub(r'<br\s*/?>', '\n', html_content)
            text = re.sub(r'</(p|li|h4)>', '\n', text)
            text = re.sub(r'<[^>]+>', '', text)

        for paragraph in text.split('\n'):
            cleaned_paragraph = " ".join(paragraph.strip().split())
            if not cleaned_paragraph:
                continue
            
            is_list_item = cleaned_paragraph.startswith(("•", "-"))
            prefix = "• " if is_list_item else ""
            if is_list_item:
                cleaned_paragraph = cleaned_paragraph[1:].strip()

            if prefix:
                self.cell(5, 5, prefix)
                self.multi_cell(0, 5, cleaned_paragraph, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
            else:
                self.multi_cell(0, 5, cleaned_paragraph, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
            self.ln(2)

    def add_pandas_table(self, df: pd.DataFrame):
        self.set_font('Helvetica', 'B', 9)
        available_width = self.w - self.l_margin - self.r_margin
        col_widths = [available_width * w for w in [0.28, 0.24, 0.24, 0.24]]
        line_height = 8

        # Encabezados de la tabla
        self.set_fill_color(*hex_to_rgb(PALETA_COLORES['primario']))
        self.set_text_color(*hex_to_rgb(PALETA_COLORES['texto_claro']))
        for i, header in enumerate(df.columns):
            self.cell(col_widths[i], line_height, header, border=0, align='C', fill=True)
        self.ln(line_height)

        # Filas de datos
        self.set_font('Helvetica', '', 9)
        self.set_text_color(*hex_to_rgb(PALETA_COLORES['texto_principal']))
        for index, row in df.iterrows():
            is_total_row = 'TOTAL' in row.get('Modalidad', '')
            
            # Alternar color de fondo para las filas
            fill_color = hex_to_rgb('#F8F9FA') if index % 2 == 0 else hex_to_rgb(PALETA_COLORES['fondo_contenedor'])
            if is_total_row:
                self.set_font('Helvetica', 'B', 9.5)
                fill_color = hex_to_rgb('#E9ECEF')

            self.set_fill_color(*fill_color)
            for i, datum in enumerate(row):
                align = 'L' if i == 0 else 'C'
                self.cell(col_widths[i], line_height, str(datum), border='T', align=align, fill=True)
            self.ln(line_height)

            if is_total_row:
                self.set_font('Helvetica', '', 9)
        self.ln(5)

    def add_image_from_buffer(self, img_buffer: io.BytesIO, title: str):
        if title:
            self.chapter_title(title)
        img_buffer.seek(0)
        # Se calcula el ancho de la imagen para que ocupe el ancho disponible de la página.
        img_width = self.w - self.l_margin - self.r_margin
        self.image(img_buffer, w=img_width)
        plt.close('all') # Cierra todas las figuras de matplotlib para liberar memoria.
        self.ln(5)

# ==============================================================================
# LÓGICA DE CÁLCULO (Sin cambios funcionales)
# ==============================================================================
class LogicaCalculo:
    @staticmethod
    def _calcular_reserva_individual(vacantes: int) -> Tuple[int, EstadoCalculo]:
        if vacantes <= 0:
            return 0, EstadoCalculo.CERO_VACANTES
        if vacantes == 1:
            return 0, EstadoCalculo.AJUSTE_V1
        return math.ceil(vacantes * 0.07), EstadoCalculo.NORMAL

    @staticmethod
    def determinar_resultados_finales(datos_entrada: DatosEntrada) -> ResultadosSimulacion:
        v_ingreso, v_ascenso = datos_entrada.v_ingreso, datos_entrada.v_ascenso
        r_ing, e_ing = LogicaCalculo._calcular_reserva_individual(v_ingreso)

        if v_ascenso > 0 and not datos_entrada.hay_pcd_para_ascenso:
            r_asc, e_asc = 0, EstadoCalculo.AJUSTE_SIN_PCD
        else:
            r_asc, e_asc = LogicaCalculo._calcular_reserva_individual(v_ascenso)

        ingreso = ModalidadResultados(
            total=v_ingreso, reserva=r_ing, general=max(0, v_ingreso - r_ing), estado=e_ing
        )
        ascenso = ModalidadResultados(
            total=v_ascenso, reserva=r_asc, general=max(0, v_ascenso - r_asc), estado=e_asc
        )
        return ResultadosSimulacion(ingreso=ingreso, ascenso=ascenso)

# ==============================================================================
# CLASE GENERADORA DE REPORTES (Con rediseño visual)
# ==============================================================================
class GeneradorReporte:
    def __init__(self, nombre_entidad: str, datos_entrada: DatosEntrada, resultados: ResultadosSimulacion):
        self.nombre_entidad = nombre_entidad
        self.datos_entrada = datos_entrada
        self.resultados = resultados
        self.total_opec = datos_entrada.total_opec
        # El gráfico se genera una sola vez en la inicialización para reutilizarlo.
        self.grafico_principal_buffer = self.crear_grafico_barras_apiladas()

    def _calcular_porcentaje_str(self, valor: float, total: float) -> str:
        return "0.0%" if total <= 0 else f"{(valor / total) * 100:.1f}%"

    def _preparar_datos_tabla(self) -> pd.DataFrame:
        i, a = self.resultados.ingreso, self.resultados.ascenso
        tr = i.reserva + a.reserva
        tg = i.general + a.general

        reserva_ing_str = f"{i.reserva} ({self._calcular_porcentaje_str(i.reserva, i.total)})" if i.total > 0 else f"{i.reserva}"
        reserva_asc_str = f"{a.reserva} ({self._calcular_porcentaje_str(a.reserva, a.total)})" if a.total > 0 else f"{a.reserva}"

        return pd.DataFrame({
            'Modalidad': ['Ingreso', 'Ascenso', 'TOTAL'],
            'Vacantes Totales': [i.total, a.total, self.total_opec],
            'Reserva PcD (7%)': [reserva_ing_str, reserva_asc_str, tr],
            'Concurso General': [i.general, a.general, tg]
        })

    def generar_tabla_html(self) -> str:
        df = self._preparar_datos_tabla().astype(str)
        
        # NOTA DE DISEÑO: Estilos de la tabla en HTML.
        # Se usan los colores de la paleta para mantener la consistencia visual.
        # La última fila (TOTAL) se resalta en negrita.
        df.iloc[-1] = df.iloc[-1].apply(lambda x: f"<strong>{x}</strong>")
        styles = [
            dict(selector="th", props=[
                ("font-size", "11pt"), ("text-align", "center"),
                ("background-color", PALETA_COLORES['primario']),
                ("color", PALETA_COLORES['texto_claro']),
                ("padding", "12px"), ("border-radius", "6px 6px 0 0")
            ]),
            dict(selector="td", props=[
                ("font-size", "10.5pt"), ("text-align", "center"),
                ("border", f"1px solid {PALETA_COLORES['borde']}"),
                ("color", PALETA_COLORES['texto_principal']),
                ("padding", "10px")
            ]),
            dict(selector="tr", props=[("background-color", PALETA_COLORES['fondo_contenedor'])]),
            dict(selector="tr:nth-child(even)", props=[("background-color", "#F8F9FA")]),
            dict(selector="tr:hover", props=[("background-color", "#E9ECEF")])
        ]
        return df.style.set_table_styles(styles).hide(axis="index").to_html(escape=False)

    def _generar_mensajes_base(self) -> List[str]:
        mensajes: List[str] = []
        r = self.resultados

        if r.ascenso.estado == EstadoCalculo.AJUSTE_SIN_PCD:
            mensajes.append(f"<li><strong>Ajuste en Ascenso:</strong> Se indicó que no existen servidores que cumplan los requisitos para la modalidad de ascenso, por lo que la reserva se ajusta a <strong>0</strong>.</li>")
        for nombre, datos in [('INGRESO', r.ingreso), ('ASCENSO', r.ascenso)]:
            if datos.estado == EstadoCalculo.AJUSTE_V1:
                mensajes.append(f"<li><strong>Nota ({nombre}):</strong> Con solo <strong>1 vacante</strong>, no se aplica reserva según la normativa.</li>")
        if r.ascenso.reserva > 0:
            mensajes.append(f"<li><strong>Nota Importante sobre Ascenso:</strong> La reserva de <strong>{r.ascenso.reserva}</strong> vacante(s) está condicionada a la existencia de <strong>servidores con derechos de carrera administrativa</strong> que, a su vez, tengan <strong>discapacidad certificada</strong> y cumplan los demás requisitos.</li>")

        if not mensajes and self.total_opec > 0:
            mensajes.append("<li>Cálculo realizado sin advertencias especiales.</li>")
        elif self.total_opec == 0:
            mensajes.append("<li>No hay vacantes para calcular.</li>")
        return mensajes

    def generar_mensajes_html(self) -> str:
        contenido = "".join(self._generar_mensajes_base())
        return f"<ul style='padding-left:20px;font-size:0.95em;line-height:1.6;color:{PALETA_COLORES['texto_secundario']};'>{contenido}</ul>"

    def _generar_conclusion_base(self) -> str:
        if self.total_opec == 0: return ""
        return (
            f"<h4 style='margin-top:15px; margin-bottom:10px; color:{PALETA_COLORES['secundario']};'>Pasos Siguientes y Consideraciones Clave:</h4>"
            f"<ul style='padding-left:20px;font-size:0.9em; line-height:1.6;color:{PALETA_COLORES['texto_secundario']};'>"
            "<li><strong>Representatividad Jerárquica:</strong> Se debe procurar que la reserva de empleos refleje la diversidad de los niveles jerárquicos de la entidad.</li>"
            "<li><strong>Análisis de Empleos:</strong> Las vacantes seleccionadas para la reserva deben ser objeto de un estudio que incluya el análisis de funciones y los ajustes razonables.</li>"
            "<li><strong>Uso del \"Recomendador de Empleos PcD\":</strong> Se invita a la entidad a usar la herramienta complementaria de la CNSC.</li>"
            "<li><strong>Validación Profesional:</strong> Los resultados deben ser validados por un profesional en Salud y Seguridad en el Trabajo (SST) o por la ARL.</li>"
            "</ul>"
        )

    def _render_fig_to_buffer(self, fig: plt.Figure) -> io.BytesIO:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', transparent=True)
        buf.seek(0)
        plt.close(fig)
        return buf

    def crear_grafico_barras_apiladas(self) -> Optional[io.BytesIO]:
        if self.total_opec == 0: return None

        res = self.resultados
        labels = ['Ascenso', 'Ingreso']
        general_data = [res.ascenso.general, res.ingreso.general]
        reserva_data = [res.ascenso.reserva, res.ingreso.reserva]

        # NOTA DE DISEÑO: Gráfico de Matplotlib.
        # Se usa un fondo transparente para que se integre con el fondo del contenedor HTML.
        # Los colores y fuentes coinciden con la paleta principal.
        fig, ax = plt.subplots(figsize=(10, 3), facecolor=PALETA_COLORES['fondo_contenedor'])
        fig.patch.set_alpha(0.0) # Fondo de la figura transparente
        ax.set_facecolor(PALETA_COLORES['fondo_contenedor'])
        ax.patch.set_alpha(0.0) # Fondo del eje transparente

        bars1 = ax.barh(labels, general_data, color=PALETA_COLORES['grafico_general'], label='General', height=0.6)
        bars2 = ax.barh(labels, reserva_data, left=general_data, color=PALETA_COLORES['grafico_reserva'], label='Reserva PcD', height=0.6)

        # Añadir etiquetas de texto dentro de las barras.
        for bar_group, color_hex in [(bars1, PALETA_COLORES['grafico_general']), (bars2, PALETA_COLORES['grafico_reserva'])]:
            for bar in bar_group:
                width = bar.get_width()
                if width > 0:
                    r, g, b = hex_to_rgb(color_hex)
                    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
                    text_color = PALETA_COLORES['texto_principal'] if luminance > 0.6 else PALETA_COLORES['texto_claro']
                    ax.text(bar.get_x() + width / 2, bar.get_y() + bar.get_height() / 2, f'{width}',
                            ha='center', va='center', fontsize=12, weight='bold', color=text_color)

        # Limpiar ejes y bordes para un look minimalista.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(PALETA_COLORES['borde'])
        ax.spines['bottom'].set_color(PALETA_COLORES['borde'])
        ax.tick_params(axis='x', colors=PALETA_COLORES['texto_secundario'])
        ax.tick_params(axis='y', colors=PALETA_COLORES['texto_principal'], length=0)
        ax.set_yticklabels(labels, fontsize=12, weight='bold')
        ax.set_xlabel(f'Total de Vacantes: {self.total_opec}', fontsize=12, labelpad=15, color=PALETA_COLORES['texto_principal'])

        legend_patches = [
            mpatches.Patch(color=PALETA_COLORES['grafico_general'], label='Vacantes Generales'),
            mpatches.Patch(color=PALETA_COLORES['grafico_reserva'], label='Vacantes Reserva PcD')
        ]
        ax.legend(handles=legend_patches, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.4), frameon=False, fontsize=11)

        plt.tight_layout(pad=1.5)
        return self._render_fig_to_buffer(fig)

    def get_reporte_html(self) -> str:
        from base64 import b64encode
        def img_to_base64_html(b: Optional[io.BytesIO]) -> str:
            if not b: return ""
            data = b64encode(b.getvalue()).decode("utf-8")
            return f'<img src="data:image/png;base64,{data}" style="width:100%;max-width:700px;margin:15px auto;display:block;"/>'

        # NOTA DE DISEÑO: Estructura del reporte en HTML.
        # Se usa un diseño de "tarjetas" (cards) para separar visualmente cada sección del reporte.
        # Las sombras y bordes redondeados le dan un aspecto moderno.
        grafico_html = img_to_base64_html(self.grafico_principal_buffer)
        
        def create_card(title: str, content: str, icon: str) -> str:
            return (
                f'<div class="report-card">'
                f'<h3>{icon} {title}</h3>'
                f'<div class="card-content">{content}</div>'
                f'</div>'
            )

        cards_html = [
            create_card("Distribución Gráfica de Vacantes", grafico_html, "📊"),
            create_card("Resumen Numérico", f'<div style="overflow-x:auto;">{self.generar_tabla_html()}</div>', "🔢"),
            create_card("Notas y Advertencias", self.generar_mensajes_html(), "⚠️"),
            create_card("Conclusión y Pasos Siguientes", self._generar_conclusion_base(), "💡")
        ]

        return f"""
        <style>
            .report-container {{
                font-family: sans-serif; border: 1px solid {PALETA_COLORES['borde']};
                border-radius: 12px; padding: 25px; background: {PALETA_COLORES['fondo_contenedor']};
                box-shadow: 0 6px 12px rgba(0,0,0,0.08); color: {PALETA_COLORES['texto_principal']};
            }}
            .report-header {{
                color: {PALETA_COLORES['primario']}; border-bottom: 3px solid {PALETA_COLORES['acento']};
                padding-bottom: 15px; text-align: center; margin-bottom: 25px;
            }}
            .report-card {{
                background: #FDFDFD; border: 1px solid {PALETA_COLORES['borde']};
                border-radius: 8px; padding: 20px; margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            .report-card h3 {{
                color: {PALETA_COLORES['secundario']}; margin-top: 0;
                border-bottom: 1px solid {PALETA_COLORES['borde']}; padding-bottom: 10px;
            }}
        </style>
        <div class="report-container">
            <h1 class="report-header">Reporte de Simulación: {self.nombre_entidad}</h1>
            {''.join(cards_html)}
        </div>
        """

    def generar_pdf_en_memoria(self) -> Tuple[str, bytes]:
        pdf = PDF_Reporte()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.alias_nb_pages()
        pdf.add_page()

        fecha_generado = datetime.now(BOGOTA_TZ).strftime('%d/%m/%Y %H:%M:%S %Z') if BOGOTA_TZ else datetime.now().strftime('%d/%m/%Y %H:%M:%S')

        pdf.set_font('Helvetica', 'B', 18)
        pdf.set_text_color(*hex_to_rgb(PALETA_COLORES['primario']))
        pdf.cell(0, 10, 'Reporte de Simulación de Vacantes OPEC', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.set_font('Helvetica', '', 14)
        pdf.set_text_color(*hex_to_rgb(PALETA_COLORES['texto_secundario']))
        pdf.cell(0, 8, self.nombre_entidad, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.set_font('Helvetica', 'I', 9)
        pdf.cell(0, 6, f"Generado: {fecha_generado}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)

        pdf.chapter_title('Parámetros de la Simulación')
        params_html = (
            f"- Total Vacantes OPEC: <strong>{self.total_opec}</strong><br/>"
            f"- Opción de Cálculo: {self.datos_entrada.opcion_calculo_str}<br/>"
            f"- Vacantes Ingreso: {self.datos_entrada.v_ingreso}<br/>"
            f"- Vacantes Ascenso: {self.datos_entrada.v_ascenso}"
        )
        if self.datos_entrada.v_ascenso > 0:
            pcd_ascenso_str = 'Sí' if self.datos_entrada.hay_pcd_para_ascenso else 'No'
            params_html += f"<br/>- Existen servidores elegibles para ascenso?: {pcd_ascenso_str}"
        pdf.chapter_body_html(params_html)

        pdf.chapter_title('Resultados Numéricos')
        pdf.add_pandas_table(self._preparar_datos_tabla())

        if self.total_opec > 0 and self.grafico_principal_buffer:
            pdf.add_image_from_buffer(self.grafico_principal_buffer, "Distribución Gráfica de Vacantes")

        if pdf.get_y() > 190: pdf.add_page()
        pdf.chapter_title('Notas y Advertencias del Cálculo')
        pdf.chapter_body_html(self.generar_mensajes_html())

        if pdf.get_y() > 220: pdf.add_page()
        pdf.chapter_title('Conclusión y Pasos Siguientes')
        pdf.chapter_body_html(self._generar_conclusion_base())

        filename = f"Reporte_OPEC_{''.join(c for c in self.nombre_entidad if c.isalnum())[:25]}_{datetime.now().strftime('%Y%m%d')}.pdf"
        try:
            return filename, bytes(pdf.output())
        except (FPDFException, Exception) as e:
            st.error(f"Ocurrió un error al generar el PDF: {e}")
            return "error.pdf", b""

# ==============================================================================
# INTERFAZ DE USUARIO CON STREAMLIT (Con rediseño visual)
# ==============================================================================
def main():
    st.set_page_config(page_title="Simulador Reserva de Plazas PcD", page_icon="♿", layout="wide")

    # --- NOTA DE DISEÑO: ESTILOS CSS PERSONALIZADOS ---
    # Este bloque inyecta CSS para sobreescribir los estilos por defecto de Streamlit.
    # Se utiliza la paleta de colores definida para lograr una apariencia consistente y moderna.
    st.markdown(f"""
    <style>
        /* --- ESTILOS GENERALES --- */
        .stApp {{
            background-color: {PALETA_COLORES['fondo_app']};
            color: {PALETA_COLORES['texto_principal']};
        }}
        h1, h2, h3 {{ color: {PALETA_COLORES['primario']}; }}
        .stDivider div {{
            background-color: {PALETA_COLORES['acento']};
            height: 2px;
        }}
        
        /* --- ESTILOS PARA WIDGETS --- */
        .stButton > button {{
            background-color: {PALETA_COLORES['primario']};
            color: {PALETA_COLORES['texto_claro']};
            border-radius: 8px; padding: 12px 24px; font-weight: bold;
            border: none; transition: all 0.3s ease;
        }}
        .stButton > button:hover {{
            background-color: {PALETA_COLORES['secundario']};
            transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        .stDownloadButton > button {{
             background-color: {PALETA_COLORES['exito']};
        }}
        .stDownloadButton > button:hover {{
             background-color: {PALETA_COLORES['secundario']};
        }}
        .stTextInput input, .stNumberInput input {{
            background-color: {PALETA_COLORES['fondo_contenedor']};
            border: 1px solid {PALETA_COLORES['borde']};
            border-radius: 8px; padding: 10px;
        }}
        .stRadio > div {{ flex-direction: row; gap: 25px; }}
        
        /* --- ESTILOS PARA CONTENEDORES --- */
        .st-emotion-cache-1r6slb0, /* Contenedor principal del formulario */
        [data-testid="stExpander"] {{
            background-color: {PALETA_COLORES['fondo_contenedor']};
            border: 1px solid {PALETA_COLORES['borde']};
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        }}
        [data-testid="stMetric"] {{
            background-color: {PALETA_COLORES['fondo_app']};
            border: 1px solid {PALETA_COLORES['borde']};
            border-radius: 8px; padding: 15px;
        }}
        [data-testid="stMetricLabel"] {{
             font-weight: bold;
        }}
    </style>
    """, unsafe_allow_html=True)

    # --- ENCABEZADO ---
    col1, col2 = st.columns([1, 6])
    with col1:
        try: st.image("logo.jpg", width=120)
        except Exception: st.info("Logo no encontrado.")
    with col2:
        st.title("Simulador de Reserva de Plazas para Personas con Discapacidad")
        st.markdown(f"Herramienta para calcular la reserva legal de empleos según la OPEC, con base en la **Ley 2418 de 2024**.")
    st.divider()

    # --- FORMULARIO DE ENTRADA ---
    with st.container(border=True):
        st.header("1. Parámetros de la Simulación")
        
        col1, col2 = st.columns(2)
        nombre_entidad = col1.text_input("Nombre de la Entidad", placeholder="Ej: Comisión Nacional del Servicio Civil", help="Ingrese el nombre completo de la entidad para el reporte.")
        total_vacantes = col2.number_input("Total de Vacantes en la OPEC", min_value=0, value=100, step=1, help="Número total de vacantes disponibles en la OPEC.")

        st.subheader("2. Distribución de Vacantes (Ingreso vs. Ascenso)")
        distribucion_tipo = st.radio("Método de Distribución", ['Automático (70% Ingreso / 30% Ascenso)', 'Manual'], horizontal=True)

        es_automatico = distribucion_tipo.startswith('Automático')
        default_ascenso = round(total_vacantes * 0.3)
        
        col_ascenso, col_distribucion = st.columns([1, 2])
        with col_ascenso:
            if es_automatico:
                vacantes_ascenso = st.number_input("Vacantes para Ascenso", value=default_ascenso, disabled=True, help="Se calcula automáticamente como el 30% del total.")
            else:
                if 'vacantes_ascenso_manual' not in st.session_state: st.session_state.vacantes_ascenso_manual = default_ascenso
                st.session_state.vacantes_ascenso_manual = min(st.session_state.vacantes_ascenso_manual, total_vacantes)
                vacantes_ascenso = st.number_input("Vacantes para Ascenso", min_value=0, max_value=total_vacantes, value=st.session_state.vacantes_ascenso_manual, step=1, help="Digite el número de vacantes para ascenso.")
                st.session_state.vacantes_ascenso_manual = vacantes_ascenso
        
        vacantes_ingreso = total_vacantes - vacantes_ascenso
        with col_distribucion:
            st.metric(label="Distribución Calculada", value=f"{vacantes_ingreso} para Ingreso", delta=f"{vacantes_ascenso} para Ascenso", delta_color="off")

        st.subheader("3. Condición para Reserva en Ascenso")
        if vacantes_ascenso > 0:
            respuesta_elegibilidad = st.radio(
                "¿Existen servidores con derechos de carrera y discapacidad que cumplen los requisitos para los cargos de ascenso?",
                ['Sí, existen servidores que cumplen los requisitos', 'No, no existen servidores que cumplen los requisitos'],
                help="Esta respuesta es crucial para determinar si se aplica la reserva del 7% a las vacantes de ascenso."
            )
            pcd_para_ascenso = respuesta_elegibilidad.startswith('Sí')
        else:
            st.info("No hay vacantes de ascenso, por lo tanto no aplica esta condición.")
            pcd_para_ascenso = False

        st.divider()

        # --- BOTÓN DE ACCIÓN ---
        if st.button("🚀 Generar Simulación y Reporte", use_container_width=True, type="primary"):
            if not nombre_entidad.strip():
                st.error("⚠️ **Error:** El nombre de la entidad es obligatorio.")
            elif vacantes_ingreso < 0:
                st.error("⚠️ **Error:** El número de vacantes de ingreso no puede ser negativo. Verifique la distribución.")
            else:
                with st.spinner("⚙️ Procesando simulación y generando reporte..."):
                    datos_entrada = DatosEntrada(
                        total_opec=total_vacantes, v_ingreso=vacantes_ingreso, v_ascenso=vacantes_ascenso,
                        opcion_calculo_str=distribucion_tipo, hay_pcd_para_ascenso=pcd_para_ascenso
                    )
                    resultados_sim = LogicaCalculo.determinar_resultados_finales(datos_entrada)
                    reporte = GeneradorReporte(nombre_entidad.strip(), datos_entrada, resultados_sim)
                
                st.success("¡Simulación completada con éxito!")
                st.markdown(reporte.get_reporte_html(), unsafe_allow_html=True)

                pdf_filename, pdf_bytes = reporte.generar_pdf_en_memoria()
                if pdf_bytes:
                    st.download_button(
                        label="📄 Descargar Reporte Completo en PDF", data=pdf_bytes,
                        file_name=pdf_filename, mime="application/pdf", use_container_width=True
                    )

    # --- PIE DE PÁGINA ---
    st.divider()
    with st.expander("Marco Normativo y Créditos"):
        st.markdown(f"""
            - **Ley 2418 de 2024:** [Consulte la norma en Función Pública](https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=249256)
            - **Circular Externa CNSC:** [Vea la circular sobre el reporte de vacantes](https://www.cnsc.gov.co/sites/default/files/2025-02/circular-externa-2025rs011333-reportede-vacantes-definitivas-aplicacion-ley-2418-2024.pdf)
            
            ---
            
            **Acerca de este Simulador:**
            
            {CREDITOS_SIMULADOR.replace("\n", "\n\n")}
        """)

if __name__ == '__main__':
    main()
