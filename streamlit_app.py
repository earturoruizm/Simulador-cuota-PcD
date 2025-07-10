# ==============================================================================
# CÓDIGO DEL SIMULADOR v15.1 (VERSIÓN CON MEJORAS ESTÉTICAS)
# Código original de: Edwin Arturo Ruiz Moreno - Comisionado Nacional del Servicio Civil
# Derechos Reservados CNSC © 2025
# Adaptación y mejoras estéticas por: Asistente de IA de Google
# ==============================================================================

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

# --- CONFIGURACIÓN E IMPORTACIONES ---
try:
    from pytz import timezone
    BOGOTA_TZ = timezone('America/Bogota')
except ImportError:
    BOGOTA_TZ = None

from fpdf import FPDF, FPDFException
from fpdf.enums import XPos, YPos

BS4_AVAILABLE = False
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    pass

# --- CONSTANTES Y ESTILOS (PALETA VISUAL BASE) ---
mpl.rcParams['figure.dpi'] = 150
pd.set_option('display.float_format', lambda x: f'{x:.1f}')
PALETA_COLORES = {
    'ingreso_general': '#B2EBF2', 
    'ingreso_reserva': '#00838F', 
    'ascenso_general': '#FFECB3',
    'ascenso_reserva': '#FF8F00', 
    'texto_claro': '#FFFFFF', 
    'texto_oscuro': '#333333',
    'fondo_titulo': '#004D40', 
    'fondo_hover': '#E0F2F1', 
    'primario': '#00796B', 
    'acento': '#FFC107',
    'fondo_app': '#F5F5F5', # Añadido para fondo general
    'borde': '#E0E0E0'      # Añadido para bordes sutiles
}
CREDITOS_SIMULADOR = (
    "Código original de: Edwin Arturo Ruiz Moreno - Comisionado Nacional del Servicio Civil\n"
    "Derechos Reservados CNSC © 2025"
)

# ==============================================================================
# CLASES DE LÓGICA DE NEGOCIO
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
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


class PDF_Reporte(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}}', align='R')
        self.set_y(-18)
        self.set_x(self.l_margin)
        self.set_font('Helvetica', 'I', 7)
        for line in CREDITOS_SIMULADOR.replace("©", "(c)").split('\n'):
            self.multi_cell(0, 4, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.set_text_color(0, 0, 0)

    def chapter_title(self, text: str):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(*hex_to_rgb(PALETA_COLORES['primario']))
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.set_draw_color(*hex_to_rgb(PALETA_COLORES['acento']))
        self.line(self.get_x(), self.get_y(), self.get_x() + 40, self.get_y())
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def chapter_body_html(self, html_content: str):
        self.set_font('Helvetica', '', 9)
        if BS4_AVAILABLE:
            text = BeautifulSoup(html_content.replace("</li>", "\n"), "html.parser") \
                    .get_text(separator=" ")
        else:
            import re
            text = re.sub(r'<br\s*/?>', '\n', html_content)
            text = re.sub(r'</(p|li|h4)>', '\n', text)
            text = re.sub(r'<[^>]+>', '', text)

        for paragraph in text.split('\n'):
            cleaned_paragraph = " ".join(paragraph.strip().split())
            if not cleaned_paragraph:
                continue

            style = 'B' if "Pasos Siguientes" in cleaned_paragraph else ''
            is_list_item = cleaned_paragraph.startswith(("•", "-"))
            prefix = "- " if is_list_item else ""
            if is_list_item:
                cleaned_paragraph = cleaned_paragraph[1:].strip()

            self.set_font('Helvetica', style, 9)
            if prefix:
                self.cell(4, 5, prefix)
                self.multi_cell(0, 5, cleaned_paragraph,
                                new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
            else:
                self.multi_cell(0, 5, cleaned_paragraph,
                                new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
            self.ln(2)

    def add_pandas_table(self, df: pd.DataFrame):
        self.set_font('Helvetica', 'B', 8)
        available_width = self.w - self.l_margin - self.r_margin
        col_widths = [available_width * w for w in [0.25, 0.22, 0.28, 0.25]]
        line_height = 7

        # Encabezados
        self.set_fill_color(*hex_to_rgb(PALETA_COLORES['fondo_titulo']))
        self.set_text_color(*hex_to_rgb(PALETA_COLORES['texto_claro']))
        for i, header in enumerate(df.columns):
            self.cell(col_widths[i], line_height, header, border=0, align='C', fill=True)
        self.ln(line_height)

        # Filas de datos
        self.set_font('Helvetica', '', 8)
        self.set_text_color(*hex_to_rgb(PALETA_COLORES['texto_oscuro']))
        for _, row in df.iterrows():
            is_total_row = 'TOTAL' in row['Modalidad']
            if is_total_row:
                self.set_font('Helvetica', 'B', 8.5)
                self.set_fill_color(*hex_to_rgb(PALETA_COLORES['fondo_hover']))
            else:
                self.set_fill_color(255, 255, 255)

            for i, datum in enumerate(row):
                align = 'L' if i == 0 else 'C'
                self.cell(col_widths[i], line_height, str(datum), border='T', align=align, fill=True)
            self.ln(line_height)

            if is_total_row:
                self.set_font('Helvetica', '', 8)
        self.ln(4)

    def add_image_from_buffer(self, img_buffer: io.BytesIO, title: str):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(*hex_to_rgb(PALETA_COLORES['primario']))
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.ln(2)
        self.set_text_color(0, 0, 0)
        img_buffer.seek(0)
        self.image(img_buffer, w=self.w - self.l_margin - self.r_margin)
        plt.close('all')
        self.ln(5)


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
            total=v_ingreso,
            reserva=r_ing,
            general=max(0, v_ingreso - r_ing),
            estado=e_ing
        )
        ascenso = ModalidadResultados(
            total=v_ascenso,
            reserva=r_asc,
            general=max(0, v_ascenso - r_asc),
            estado=e_asc
        )
        return ResultadosSimulacion(ingreso=ingreso, ascenso=ascenso)


class GeneradorReporte:
    def __init__(self, nombre_entidad: str, datos_entrada: DatosEntrada, resultados: ResultadosSimulacion):
        self.nombre_entidad = nombre_entidad
        self.datos_entrada = datos_entrada
        self.resultados = resultados
        self.total_opec = datos_entrada.total_opec
        self.grafico_principal_buffer = self.crear_grafico_barras_apiladas()

    def _calcular_porcentaje_str(self, valor: float, total: float) -> str:
        return "0.0%" if total <= 0 else f"{(valor / total) * 100:.1f}%"

    def _preparar_datos_tabla(self) -> pd.DataFrame:
        i, a = self.resultados.ingreso, self.resultados.ascenso
        tr = i.reserva + a.reserva
        tg = i.general + a.general

        reserva_ing_str = (
            f"{i.reserva} ({self._calcular_porcentaje_str(i.reserva, i.total)})"
            if i.total > 0 else f"{i.reserva}"
        )
        reserva_asc_str = (
            f"{a.reserva} ({self._calcular_porcentaje_str(a.reserva, a.total)})"
            if a.total > 0 else f"{a.reserva}"
        )

        return pd.DataFrame({
            'Modalidad': ['Ingreso', 'Ascenso', 'TOTAL'],
            'Vacantes Totales': [i.total, a.total, self.total_opec],
            'Reserva PcD': [reserva_ing_str, reserva_asc_str, tr],
            'Concurso General': [i.general, a.general, tg]
        })

    def generar_tabla_html(self) -> str:
        df_styled = self._preparar_datos_tabla().astype(str)
        df_styled.iloc[-1] = df_styled.iloc[-1].apply(lambda x: f"<strong>{x}</strong>")

        styles = [
            dict(selector="th", props=[
                ("font-size", "11pt"),
                ("text-align", "center"),
                ("background-color", PALETA_COLORES['fondo_titulo']),
                ("color", PALETA_COLORES['texto_claro']),
                ("padding", "8px")
            ]),
            dict(selector="td", props=[
                ("font-size", "10.5pt"),
                ("text-align", "center"),
                ("border", f"1px solid {PALETA_COLORES['borde']}"),
                ("color", PALETA_COLORES['texto_oscuro']),
                ("background-color", "#FFFFFF"),
                ("padding", "8px")
            ])
        ]
        return (
            df_styled
            .style
            .set_table_styles(styles)
            .hide(axis="index")
            .to_html(escape=False)
        )

    def _generar_mensajes_base(self) -> List[str]:
        mensajes: List[str] = []
        r = self.resultados

        if r.ascenso.estado == EstadoCalculo.AJUSTE_SIN_PCD:
            mensajes.append(
                "<li><strong>Ajuste en Ascenso:</strong> Se indicó que no existen servidores que cumplan "
                "los requisitos para la modalidad de ascenso, por lo que la reserva se ajusta a <strong>0</strong>.</li>"
            )

        for nombre, datos in [('INGRESO', r.ingreso), ('ASCENSO', r.ascenso)]:
            if datos.estado == EstadoCalculo.AJUSTE_V1:
                mensajes.append(
                    f"<li><strong>Nota ({nombre}):</strong> Con solo <strong>1 vacante</strong>, no se aplica reserva.</li>"
                )

        if r.ascenso.reserva > 0:
            mensajes.append(
                "<li><strong>Nota Importante sobre Ascenso:</strong> La reserva de "
                f"<strong>{r.ascenso.reserva}</strong> vacante(s) está condicionada a la existencia de "
                "<strong>servidores con derechos de carrera administrativa</strong> y, a su vez, tengan "
                "<strong>discapacidad certificada</strong> y cumplan los demás requisitos.</li>"
            )

        if not mensajes and self.total_opec > 0:
            mensajes.append("<li>No se generaron advertencias especiales.</li>")
        elif self.total_opec == 0:
            mensajes.append("<li>No hay vacantes para calcular.</li>")

        return mensajes

    def generar_mensajes_html(self) -> str:
        contenido = "".join(self._generar_mensajes_base())
        return (
            f"<ul style='padding-left:20px;font-size:0.95em;"
            f"color:{PALETA_COLORES['texto_oscuro']};'>{contenido}</ul>"
        )

    def _generar_conclusion_base(self) -> str:
        if self.total_opec == 0:
            return ""
        return (
            f"<h4 style='margin-top:15px; margin-bottom:5px; color:{PALETA_COLORES['fondo_titulo']};'>"
            "Pasos Siguientes y Consideraciones Clave:</h4>"
            "<ul style='padding-left:20px;font-size:0.9em; line-height:1.6;'>"
            "<li><strong>Representatividad Jerárquica:</strong> Se debe procurar que la reserva de "
            "empleos refleje la diversidad de los niveles jerárquicos de la entidad.</li>"
            "<li><strong>Análisis de Empleos:</strong> Las vacantes seleccionadas para la reserva deben "
            "ser objeto de un estudio que incluya el análisis de funciones y los ajustes razonables.</li>"
            "<li><strong>Uso del \"Recomendador de Empleos PcD\":</strong> Se invita a la entidad a usar la "
            "herramienta complementaria de la CNSC.</li>"
            "<li><strong>Validación Profesional:</strong> Los resultados deben ser validados por un "
            "profesional en Salud y Seguridad en el Trabajo (SST) o por la ARL.</li>"
            "</ul>"
        )

    def _render_fig_to_buffer(self, fig: plt.Figure) -> io.BytesIO:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

    def crear_grafico_barras_apiladas(self) -> Optional[io.BytesIO]:
        if self.total_opec == 0:
            return None

        res = self.resultados
        labels = ['Ascenso', 'Ingreso']
        general_data = [res.ascenso.general, res.ingreso.general]
        reserva_data = [res.ascenso.reserva, res.ingreso.reserva]

        fig, ax = plt.subplots(figsize=(10, 3.5), facecolor='white')
        ax.set_facecolor('white')
        
        bars1 = ax.barh(labels, general_data, color=PALETA_COLORES['ingreso_general'], label='General')
        bars2 = ax.barh(labels, reserva_data, left=general_data, color=PALETA_COLORES['ingreso_reserva'], label='Reserva PcD')

        for bar_group, color_hex in [(bars1, PALETA_COLORES['ingreso_general']), (bars2, PALETA_COLORES['ingreso_reserva'])]:
            for bar in bar_group:
                width = bar.get_width()
                if width > 0:
                    r, g, b = hex_to_rgb(color_hex)
                    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
                    text_color = PALETA_COLORES['texto_oscuro'] if luminance > 0.5 else PALETA_COLORES['texto_claro']
                    ax.text(
                        bar.get_x() + width / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f'{int(width)}',
                        ha='center', va='center',
                        fontsize=12, weight='bold', color=text_color
                    )

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.tick_params(bottom=False, left=False)
        ax.set_xticks([])
        ax.set_yticklabels(labels, fontsize=12, weight='bold', color=PALETA_COLORES['texto_oscuro'])
        ax.set_xlabel(f"Total de Vacantes: {self.total_opec}", fontsize=12, labelpad=10, color=PALETA_COLORES['texto_oscuro'])

        legend_patches = [
            mpatches.Patch(color=PALETA_COLORES['ingreso_general'], label='Vacantes Generales'),
            mpatches.Patch(color=PALETA_COLORES['ingreso_reserva'], label='Vacantes Reserva PcD')
        ]
        ax.legend(
            handles=legend_patches,
            loc='lower center',
            ncol=2,
            bbox_to_anchor=(0.5, -0.3),
            frameon=False,
            fontsize=11
        )

        plt.tight_layout(pad=1)
        return self._render_fig_to_buffer(fig)

    def get_reporte_html(self) -> str:
        from base64 import b64encode

        def img(b: Optional[io.BytesIO]) -> str:
            if not b:
                return ""
            data = b64encode(b.getvalue()).decode("utf-8")
            return (
                f'<img src="data:image/png;base64,{data}" '
                f'style="width:100%;max-width:700px;margin:auto;display:block;border-radius:8px;"/>'
            )

        grafico_html = img(self.grafico_principal_buffer)
        return (
            f"""<div style="font-family: 'Roboto', sans-serif; border:1px solid {PALETA_COLORES['borde']};"""
            f"""border-radius:12px;padding:25px;background:#FFFFFF; box-shadow: 0 4px 12px rgba(0,0,0,0.05);"""
            f"""color:{PALETA_COLORES['texto_oscuro']};">"""
            f"""<h1 style="color:{PALETA_COLORES['fondo_titulo']};"""
            f"""border-bottom:3px solid {PALETA_COLORES['acento']};"""
            f"""padding-bottom:10px; text-align:center;">📊 Reporte de Simulación: {self.nombre_entidad}</h1>"""
            f"""<div style="margin-top:25px; padding:15px; border-radius:8px; background-color: {PALETA_COLORES['fondo_hover']};">
                   <h2 style="color:{PALETA_COLORES['primario']}; margin-top:0;">Distribución Gráfica de Vacantes</h2>
                   {grafico_html}
               </div>"""
            f"""<div style="margin-top:25px;">
                   <h2 style="color:{PALETA_COLORES['primario']};">Resumen de Distribución</h2>
                   {self.generar_tabla_html()}
               </div>"""
            f"""<div style="margin-top:25px; background:#FFFFFF; border-left:5px solid {PALETA_COLORES['acento']}; padding:1px 15px; border-radius:4px;">
                   <h2 style="color:{PALETA_COLORES['primario']};">Notas y Advertencias Clave</h2>
                   {self.generar_mensajes_html()}
               </div>"""
            f"""<div style="margin-top:25px; background:{PALETA_COLORES['fondo_hover']}; padding:15px; border-radius:8px;">
                   {self._generar_conclusion_base()}
               </div>"""
            f"""</div>"""
        )

    def generar_pdf_en_memoria(self) -> Tuple[str, bytes]:
        pdf = PDF_Reporte()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.alias_nb_pages()
        pdf.add_page()

        fecha_generado = (
            datetime.now(BOGOTA_TZ).strftime('%d/%m/%Y %H:%M:%S %Z')
            if BOGOTA_TZ else
            datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        )

        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, 'Reporte de Simulación de Vacantes OPEC',
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.set_font('Helvetica', '', 12)
        pdf.cell(0, 8, self.nombre_entidad,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.set_font('Helvetica', '', 9)
        pdf.cell(0, 6, f"Generado: {fecha_generado}",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(8)

        pdf.chapter_title('Parámetros de la Simulación')
        params_html = (
            f"- Total Vacantes OPEC: {self.total_opec}\n"
            f"- Opción de Cálculo: {self.datos_entrada.opcion_calculo_str}\n"
            f"- Vacantes Ingreso: {self.datos_entrada.v_ingreso}\n"
            f"- Vacantes Ascenso: {self.datos_entrada.v_ascenso}"
        )
        if self.datos_entrada.v_ascenso > 0:
            params_html += (
                f"\n- Existen servidores que cumplen requisitos para ascenso?: "
                f"{'Sí' if self.datos_entrada.hay_pcd_para_ascenso else 'No'}"
            )
        pdf.chapter_body_html(params_html)

        pdf.chapter_title('Resultados Numéricos')
        pdf.add_pandas_table(self._preparar_datos_tabla())

        if self.total_opec > 0:
            pdf.chapter_title("Distribución Gráfica de Vacantes")
            if buffer := self.grafico_principal_buffer:
                pdf.add_image_from_buffer(buffer, "")

        if pdf.get_y() > 200:
            pdf.add_page()

        pdf.chapter_title('Notas y Advertencias del Cálculo')
        pdf.chapter_body_html(self.generar_mensajes_html())

        pdf.chapter_title('Conclusión y Pasos Siguientes')
        pdf.chapter_body_html(self._generar_conclusion_base())

        filename = (
            f"Reporte_OPEC_{''.join(c for c in self.nombre_entidad if c.isalnum())[:30]}_"
            f"{datetime.now(BOGOTA_TZ).strftime('%Y%m%d') if BOGOTA_TZ else datetime.now().strftime('%Y%m%d')}.pdf"
        )
        try:
            return filename, bytes(pdf.output())
        except (FPDFException, Exception) as e:
            st.error(f"Ocurrió un error al generar el PDF: {e}")
            return "error.pdf", b""


# ==============================================================================
# INTERFAZ DE USUARIO Y LÓGICA PRINCIPAL
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Simulador Reserva de Plazas PcD",
        page_icon="♿",
        layout="wide"
    )

    # --- INYECCIÓN DE CSS PARA MEJORAS ESTÉTICAS ---
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
            
            body, .stApp {{
                font-family: 'Roboto', sans-serif;
                background-color: {PALETA_COLORES['fondo_app']};
            }}
            
            .stButton > button {{
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                border: 2px solid {PALETA_COLORES['primario']};
                background-color: {PALETA_COLORES['primario']};
                color: {PALETA_COLORES['texto_claro']};
                transition: all 0.3s ease-in-out;
            }}
            .stButton > button:hover {{
                background-color: {PALETA_COLORES['texto_claro']};
                color: {PALETA_COLORES['primario']};
                border-color: {PALETA_COLORES['primario']};
            }}
            
            [data-testid="stContainer"], [data-testid="stVerticalBlock"] {{
                background-color: #FFFFFF;
                border: 1px solid {PALETA_COLORES['borde']};
                border-radius: 10px;
                padding: 25px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.04);
            }}
            
            .stTextInput > div > div > input, .stNumberInput > div > div > input {{
                background-color: {PALETA_COLORES['fondo_hover']};
                border-radius: 5px;
            }}
            
            h1, h2, h3 {{ color: {PALETA_COLORES['fondo_titulo']}; }}
            st.subheader {{ color: {PALETA_COLORES['primario']}; }}
            
        </style>
    """, unsafe_allow_html=True)


    # --- ENCABEZADO ---
    col1, col2 = st.columns([1, 5])
    with col1:
        try:
            st.image("logo.jpg", width=120)
        except Exception:
            st.warning("Logo no encontrado.")
    with col2:
        st.title("Simulador de Reserva de Plazas para Personas con Discapacidad")
        st.markdown("Herramienta para calcular la reserva legal de empleos según la OPEC.")

    st.divider()

    # --- FORMULARIO DE ENTRADA ---
    with st.container(border=True):
        st.subheader("📝 1. Datos Generales")
        col1, col2 = st.columns(2)
        nombre_entidad = col1.text_input(
            "Nombre de la Entidad",
            placeholder="Ej: Alcaldía Mayor de Bogotá D.C."
        )
        total_vacantes = col2.number_input(
            "Total de Vacantes en la OPEC",
            min_value=0, value=100, step=1
        )

        st.subheader("🔀 2. Distribución de Vacantes")
        distribucion_tipo = st.radio(
            "Método",
            options=['Automático (70/30)', 'Manual'],
            horizontal=True
        )

        es_automatico = (distribucion_tipo == 'Automático (70/30)')
        default_ascenso = round(total_vacantes * 0.3)
        if es_automatico:
            vacantes_ascenso = default_ascenso
            st.number_input(
                "Vacantes para Ascenso",
                value=vacantes_ascenso,
                disabled=True,
                help="Se calcula automáticamente como el 30% del total."
            )
        else:
            if 'vacantes_ascenso_manual' not in st.session_state:
                st.session_state.vacantes_ascenso_manual = default_ascenso
            st.session_state.vacantes_ascenso_manual = min(st.session_state.vacantes_ascenso_manual, total_vacantes)
            vacantes_ascenso = st.number_input(
                "Vacantes para Ascenso",
                min_value=0,
                max_value=total_vacantes,
                value=st.session_state.vacantes_ascenso_manual,
                step=1,
                help="Digite el número de vacantes para ascenso."
            )
            st.session_state.vacantes_ascenso_manual = vacantes_ascenso

        vacantes_ingreso = total_vacantes - vacantes_ascenso
        st.metric(
            label="Distribución Calculada",
            value=f"{vacantes_ingreso} Ingreso",
            delta=f"{vacantes_ascenso} Ascenso",
            delta_color="off"
        )

        st.subheader("♿ 3. Cumplimiento de Requisitos para Ascenso")
        if vacantes_ascenso > 0:
            respuesta_elegibilidad = st.radio(
                "¿Existen servidores con derechos de carrera y discapacidad que cumplen los requisitos para los cargos de ascenso?",
                options=[
                    'Sí, existen servidores que cumplen los requisitos',
                    'No, no existen servidores que cumplen los requisitos'
                ]
            )
            pcd_para_ascenso = (respuesta_elegibilidad.startswith('Sí'))
        else:
            st.info(
                "No hay vacantes de ascenso, por lo tanto no aplica esta condición."
            )
            pcd_para_ascenso = False

        st.divider()

        # --- BOTÓN DE ACCIÓN ---
        if st.button(
            label="🚀 Generar Simulación y Reporte",
            use_container_width=True,
            type="primary"
        ):
            if not nombre_entidad.strip():
                st.error("⚠️ **Error:** El nombre de la entidad es obligatorio.")
            elif vacantes_ingreso < 0:
                st.error("⚠️ **Error:** El número de vacantes de ingreso no puede ser negativo.")
            else:
                with st.spinner("⚙️ Procesando, por favor espere..."):
                    datos_entrada = DatosEntrada(
                        total_opec=total_vacantes,
                        v_ingreso=vacantes_ingreso,
                        v_ascenso=vacantes_ascenso,
                        opcion_calculo_str=distribucion_tipo,
                        hay_pcd_para_ascenso=pcd_para_ascenso
                    )
                    resultados_sim = LogicaCalculo.determinar_resultados_finales(datos_entrada)
                    reporte = GeneradorReporte(nombre_entidad.strip(), datos_entrada, resultados_sim)

                st.success("¡Simulación completada!")
                st.markdown(reporte.get_reporte_html(), unsafe_allow_html=True)

                pdf_filename, pdf_bytes = reporte.generar_pdf_en_memoria()
                if pdf_bytes:
                    st.download_button(
                        label="📄 Descargar Reporte Completo en PDF",
                        data=pdf_bytes,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        use_container_width=True
                    )

    # --- PIE DE PÁGINA ---
    st.divider()
    with st.expander("Marco Normativo"):
        st.markdown(
            "- **Ley 2418 de 2024:** [Consulte la norma en Función Pública]"
            "(https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=249256)\n"
            "- **Circular Externa CNSC:** [Vea la circular sobre el reporte de vacantes]"
            "(https://www.cnsc.gov.co/sites/default/files/2025-02/"
            "circular-externa-2025rs011333-reportede-vacantes-definitivas-aplicacion-ley-2418-2024.pdf)"
        )
    with st.expander("Acerca de este Simulador"):
        st.info(CREDITOS_SIMULADOR.replace("\n", "\n\n"))


if __name__ == '__main__':
    main()
