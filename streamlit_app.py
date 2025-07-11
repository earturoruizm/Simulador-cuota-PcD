# ==============================================================================
# CÓDIGO DEL SIMULADOR v16 (VERSIÓN VERIFICADA Y MEJORADA)
# Código original de: Edwin Arturo Ruiz Moreno - Comisionado Nacional del Servicio Civil
# Derechos Reservados CNSC © 2025
# Adaptación, mejoras de UX/UI y enriquecimiento por: Asistente de IA de Google
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
from base64 import b64encode

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

# --- CONSTANTES Y PALETA DE COLORES MEJORADA ---
mpl.rcParams['figure.dpi'] = 180 # Mayor resolución para gráficos más nítidos
pd.set_option('display.float_format', lambda x: f'{x:.1f}')
PALETA_COLORES = {
    'primario': '#00796B',      # Verde azulado principal
    'secundario': '#004D40',    # Verde oscuro para títulos y fondos
    'acento': '#FFC107',        # Amarillo para destacar
    'fondo_claro': '#F5F5F5',   # Gris muy claro para el fondo de la app
    'fondo_tarjeta': '#FFFFFF', # Blanco para las tarjetas de contenido
    'texto_principal': '#333333', # Gris oscuro para texto
    'texto_claro': '#FFFFFF',   # Blanco para texto sobre fondos oscuros
    'reserva_ingreso': '#00838F', # Cian oscuro para reserva
    'general_ingreso': '#B2EBF2', # Cian claro para general
    'reserva_ascenso': '#FF8F00', # Naranja oscuro para reserva
    'general_ascenso': '#FFECB3', # Naranja claro para general
    'error': '#D32F2F',
    'info': '#0288D1',
}
CREDITOS_SIMULADOR = (
    "Código original de: Edwin Arturo Ruiz Moreno - Comisionado Nacional del Servicio Civil\n"
    "Derechos Reservados CNSC © 2025\n"
    "Esta versión ha sido mejorada y enriquecida por un Asistente de IA de Google para optimizar la experiencia de usuario."
)

# ==============================================================================
# CLASES DE LÓGICA DE NEGOCIO (SIN CAMBIOS EN EL CÁLCULO)
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
    """Clase para generar el reporte en PDF, adaptada a la nueva paleta visual."""
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}}', align='R')
        self.set_y(-20)
        self.set_x(self.l_margin)
        self.set_font('Helvetica', 'I', 7)
        for line in CREDITOS_SIMULADOR.replace("©", "(c)").split('\n'):
            self.multi_cell(0, 4, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.set_text_color(0, 0, 0)

    def chapter_title(self, text: str):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(*hex_to_rgb(PALETA_COLORES['primario']))
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L', border=0)
        self.set_draw_color(*hex_to_rgb(PALETA_COLORES['acento']))
        self.line(self.get_x(), self.get_y(), self.get_x() + 50, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def chapter_body_html(self, html_content: str):
        self.set_font('Helvetica', '', 10)
        # Limpieza de HTML para el PDF
        if BS4_AVAILABLE:
            text = BeautifulSoup(html_content.replace("</li>", "\n"), "html.parser").get_text(separator=" ")
        else:
            import re
            text = re.sub(r'<br\s*/?>', '\n', html_content)
            text = re.sub(r'</(p|li|h4|h5)>', '\n', text)
            text = re.sub(r'<[^>]+>', '', text)

        for paragraph in text.split('\n'):
            cleaned_paragraph = " ".join(paragraph.strip().split())
            if not cleaned_paragraph: continue

            is_list_item = cleaned_paragraph.startswith(("•", "-"))
            prefix = "  •  " if is_list_item else ""
            if is_list_item:
                cleaned_paragraph = cleaned_paragraph[1:].strip()

            self.set_font('Helvetica', '', 10)
            if prefix:
                self.cell(5, 5, prefix)
                self.multi_cell(0, 5, cleaned_paragraph, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='J')
            else:
                self.multi_cell(0, 5, cleaned_paragraph, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='J')
            self.ln(2)

    def add_pandas_table(self, df: pd.DataFrame):
        self.set_font('Helvetica', 'B', 9)
        available_width = self.w - self.l_margin - self.r_margin
        col_widths = [available_width * w for w in [0.25, 0.22, 0.28, 0.25]]
        line_height = 8

        # Encabezados
        self.set_fill_color(*hex_to_rgb(PALETA_COLORES['secundario']))
        self.set_text_color(*hex_to_rgb(PALETA_COLORES['texto_claro']))
        for i, header in enumerate(df.columns):
            self.cell(col_widths[i], line_height, header, border=1, align='C', fill=True)
        self.ln(line_height)

        # Filas
        self.set_font('Helvetica', '', 9)
        self.set_text_color(*hex_to_rgb(PALETA_COLORES['texto_principal']))
        for index, row in df.iterrows():
            is_total_row = 'TOTAL' in row['Modalidad']
            fill_color = hex_to_rgb('#E0F2F1') if is_total_row else (255, 255, 255)
            self.set_fill_color(*fill_color)
            if is_total_row: self.set_font('Helvetica', 'B', 9.5)

            for i, datum in enumerate(row):
                align = 'L' if i == 0 else 'C'
                self.cell(col_widths[i], line_height, str(datum), border=1, align=align, fill=True)
            self.ln(line_height)
            if is_total_row: self.set_font('Helvetica', '', 9)
        self.ln(5)

    def add_image_from_buffer(self, img_buffer: io.BytesIO, title: str):
        self.chapter_title(title)
        img_buffer.seek(0)
        # Centrar la imagen
        img_w = self.w - self.l_margin - self.r_margin
        self.image(img_buffer, w=img_w)
        plt.close('all')
        self.ln(5)


class LogicaCalculo:
    @staticmethod
    def _calcular_reserva_individual(vacantes: int) -> Tuple[int, EstadoCalculo]:
        if vacantes <= 0: return 0, EstadoCalculo.CERO_VACANTES
        if vacantes == 1: return 0, EstadoCalculo.AJUSTE_V1
        return math.ceil(vacantes * 0.07), EstadoCalculo.NORMAL

    @staticmethod
    def determinar_resultados_finales(datos_entrada: DatosEntrada) -> ResultadosSimulacion:
        v_ingreso, v_ascenso = datos_entrada.v_ingreso, datos_entrada.v_ascenso
        r_ing, e_ing = LogicaCalculo._calcular_reserva_individual(v_ingreso)

        if v_ascenso > 0 and not datos_entrada.hay_pcd_para_ascenso:
            r_asc, e_asc = 0, EstadoCalculo.AJUSTE_SIN_PCD
        else:
            r_asc, e_asc = LogicaCalculo._calcular_reserva_individual(v_ascenso)

        return ResultadosSimulacion(
            ingreso=ModalidadResultados(v_ingreso, r_ing, max(0, v_ingreso - r_ing), e_ing),
            ascenso=ModalidadResultados(v_ascenso, r_asc, max(0, v_ascenso - r_asc), e_asc)
        )


class GeneradorReporte:
    def __init__(self, nombre_entidad: str, datos_entrada: DatosEntrada, resultados: ResultadosSimulacion):
        self.nombre_entidad = nombre_entidad
        self.datos_entrada = datos_entrada
        self.resultados = resultados
        self.total_opec = datos_entrada.total_opec
        self.grafico_barras_buffer = self.crear_grafico_barras_apiladas()
        self.grafico_donut_buffer = self.crear_grafico_donut()

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
            'Reserva PcD (7%)': [reserva_ing_str, reserva_asc_str, f"<strong>{tr}</strong>"],
            'Concurso General': [i.general, a.general, f"<strong>{tg}</strong>"]
        })

    def generar_tabla_html(self) -> str:
        df_styled = self._preparar_datos_tabla().astype(str)
        
        styles = [
            dict(selector="th", props=[
                ("font-size", "12pt"), ("text-align", "center"),
                ("background-color", PALETA_COLORES['secundario']),
                ("color", PALETA_COLORES['texto_claro']),
                ("border", "1px solid " + PALETA_COLORES['secundario']),
                ("padding", "10px")
            ]),
            dict(selector="td", props=[
                ("font-size", "11pt"), ("text-align", "center"),
                ("border", "1px solid #ddd"),
                ("color", PALETA_COLORES['texto_principal']),
                ("padding", "8px")
            ]),
            dict(selector="tr:hover", props=[("background-color", "#f5f5f5")]),
            dict(selector="tr:last-child", props=[
                ("background-color", "#E0F2F1"),
                ("font-weight", "bold")
            ])
        ]
        return (df_styled.style.set_table_styles(styles)
                .hide(axis="index").to_html(escape=False))

    def _generar_mensajes_base(self) -> List[str]:
        mensajes: List[str] = []
        r = self.resultados

        if r.ascenso.estado == EstadoCalculo.AJUSTE_SIN_PCD:
            mensajes.append(f"<li><strong>Ajuste Clave en Ascenso:</strong> Se indicó que <strong>no existen servidores de carrera con discapacidad</strong> que cumplan los requisitos para la modalidad de ascenso. Por ley, la reserva para este grupo se ajusta a <strong>cero (0)</strong>.</li>")

        for nombre, datos in [('INGRESO', r.ingreso), ('ASCENSO', r.ascenso)]:
            if datos.estado == EstadoCalculo.AJUSTE_V1:
                mensajes.append(f"<li><strong>Nota Informativa ({nombre}):</strong> Al ofertarse una <strong>única vacante</strong> en esta modalidad, la ley estipula que no se aplica el porcentaje de reserva. La vacante se destina al concurso general.</li>")

        if r.ascenso.reserva > 0:
            mensajes.append(f"<li><strong>Condición para Reserva en Ascenso:</strong> La reserva de <strong>{r.ascenso.reserva} vacante(s)</strong> en la modalidad de ascenso está estrictamente condicionada a que existan <strong>servidores con derechos de carrera y discapacidad certificada</strong> que cumplan todos los requisitos del empleo a proveer.</li>")

        if not mensajes and self.total_opec > 0:
            mensajes.append("<li><strong>Cálculo Estándar:</strong> La simulación se ha realizado siguiendo los parámetros normativos estándar sin necesidad de ajustes especiales.</li>")
        elif self.total_opec == 0:
            mensajes.append("<li><strong>Sin Vacantes:</strong> No se han ingresado vacantes para calcular.</li>")

        return mensajes

    def generar_mensajes_html(self) -> str:
        contenido = "".join(self._generar_mensajes_base())
        return f"<ul style='padding-left:20px; font-size:1.05em; color:{PALETA_COLORES['texto_principal']};'>{contenido}</ul>"

    def _generar_conclusion_base(self) -> str:
        if self.total_opec == 0: return ""
        return (
            "<h5 style='margin-top:15px; margin-bottom:10px; color:" + PALETA_COLORES['secundario'] + ";'>"
            "Recomendaciones y Pasos Siguientes</h5>"
            "<p>Este simulador es el primer paso. Para una correcta aplicación de la reserva, la entidad debe:</p>"
            "<ul style='padding-left:20px; font-size:1em; line-height:1.7;'>"
            "<li><strong>1. Garantizar Representatividad:</strong> La selección de empleos para la reserva debe procurar reflejar la diversidad de los niveles jerárquicos (Asistencial, Técnico, Profesional, Asesor) y grados salariales de la entidad.</li>"
            "<li><strong>2. Realizar Análisis de Empleos (ATP):</strong> Las vacantes preseleccionadas para la reserva deben ser objeto de un Análisis de Puesto de Trabajo (ATP) para identificar funciones, competencias y barreras potenciales.</li>"
            "<li><strong>3. Utilizar Herramientas Complementarias:</strong> Se recomienda encarecidamente usar el <strong>\"Recomendador de Empleos para Personas con Discapacidad\"</strong>, una herramienta de la CNSC que asiste en la selección de los empleos más idóneos para la reserva.</li>"
            "<li><strong>4. Obtener Validación Profesional:</strong> Los resultados del ATP y la selección final deben ser validados por un profesional en Salud y Seguridad en el Trabajo (SST) o por la Administradora de Riesgos Laborales (ARL) para asegurar la pertinencia de los ajustes razonables.</li>"
            "</ul>"
        )

    def _render_fig_to_buffer(self, fig: plt.Figure) -> io.BytesIO:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=180, bbox_inches='tight', transparent=True)
        buf.seek(0)
        plt.close(fig)
        return buf

    def crear_grafico_barras_apiladas(self) -> Optional[io.BytesIO]:
        if self.total_opec == 0: return None
        res, labels = self.resultados, ['Ascenso', 'Ingreso']
        
        general_data = [res.ascenso.general, res.ingreso.general]
        reserva_data = [res.ascenso.reserva, res.ingreso.reserva]
        colors_general = [PALETA_COLORES['general_ascenso'], PALETA_COLORES['general_ingreso']]
        colors_reserva = [PALETA_COLORES['reserva_ascenso'], PALETA_COLORES['reserva_ingreso']]

        fig, ax = plt.subplots(figsize=(8, 3), facecolor=None)
        ax.set_facecolor('#FFFFFF00') # Fondo transparente
        
        bars1 = ax.barh(labels, general_data, color=colors_general)
        bars2 = ax.barh(labels, reserva_data, left=general_data, color=colors_reserva)

        for bar_group in (bars1, bars2):
            for bar in bar_group:
                width = bar.get_width()
                if width > 0:
                    ax.text(bar.get_x() + width / 2, bar.get_y() + bar.get_height() / 2,
                            f'{int(width)}', ha='center', va='center', fontsize=11,
                            weight='bold', color=PALETA_COLORES['texto_principal'])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#B0BEC5')
        ax.spines['bottom'].set_color('#B0BEC5')
        ax.tick_params(axis='x', colors='#78909C')
        ax.tick_params(axis='y', colors=PALETA_COLORES['texto_principal'], labelsize=12)
        ax.set_xlabel(f"Número de Vacantes", fontsize=11, color=PALETA_COLORES['texto_principal'], labelpad=10)
        plt.tight_layout(pad=1)
        return self._render_fig_to_buffer(fig)

    def crear_grafico_donut(self) -> Optional[io.BytesIO]:
        if self.total_opec == 0: return None
        i, a = self.resultados.ingreso, self.resultados.ascenso
        total_reserva = i.reserva + a.reserva
        total_general = i.general + a.general

        if self.total_opec == 0: return None

        labels = ['Reserva PcD', 'Concurso General']
        sizes = [total_reserva, total_general]
        colors = [PALETA_COLORES['primario'], PALETA_COLORES['acento']]
        
        fig, ax = plt.subplots(figsize=(4, 4), facecolor=None)
        ax.set_facecolor('#FFFFFF00')

        def func(pct, allvals):
            absolute = int(round(pct/100.*np.sum(allvals)))
            return f"{pct:.1f}%\n({absolute})"

        wedges, texts, autotexts = ax.pie(
            sizes, autopct=lambda pct: func(pct, sizes),
            startangle=90, colors=colors,
            pctdistance=0.8, wedgeprops=dict(width=0.4, edgecolor='w'))

        plt.setp(autotexts, size=11, weight="bold", color=PALETA_COLORES['texto_claro'])
        ax.axis('equal')
        return self._render_fig_to_buffer(fig)

    def get_reporte_html(self) -> str:
        def img_to_base64_html(b: Optional[io.BytesIO]) -> str:
            if not b: return ""
            data = b64encode(b.getvalue()).decode("utf-8")
            return f'<img src="data:image/png;base64,{data}" style="width:100%; max-width:100%; margin:auto; display:block;"/>'

        grafico_barras_html = img_to_base64_html(self.grafico_barras_buffer)
        grafico_donut_html = img_to_base64_html(self.grafico_donut_buffer)
        
        total_reserva = self.resultados.ingreso.reserva + self.resultados.ascenso.reserva
        porc_reserva_total = self._calcular_porcentaje_str(total_reserva, self.total_opec)

        return f"""
        <div style="font-family:sans-serif; border:1px solid #ddd; border-radius:12px; padding:25px; background:{PALETA_COLORES['fondo_tarjeta']}; color:{PALETA_COLORES['texto_principal']};">
            <h2 style="color:{PALETA_COLORES['secundario']}; border-bottom:3px solid {PALETA_COLORES['acento']}; padding-bottom:10px;">
                📊 Reporte de Simulación: {self.nombre_entidad}
            </h2>
            
            <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-top: 25px; text-align: center;">
                <div style="flex: 1; min-width: 150px; background: #E0F2F1; padding: 15px; border-radius: 8px;">
                    <h3 style="margin:0; color:{PALETA_COLORES['secundario']}; font-size: 1.1em;">Total Vacantes OPEC</h3>
                    <p style="margin:0; font-size: 2.2em; font-weight: bold;">{self.total_opec}</p>
                </div>
                <div style="flex: 1; min-width: 150px; background: #E8EAF6; padding: 15px; border-radius: 8px;">
                    <h3 style="margin:0; color:#303F9F; font-size: 1.1em;">Total Reserva PcD</h3>
                    <p style="margin:0; font-size: 2.2em; font-weight: bold;">{total_reserva}</p>
                </div>
                <div style="flex: 1; min-width: 150px; background: #FFF3E0; padding: 15px; border-radius: 8px;">
                    <h3 style="margin:0; color:#F57C00; font-size: 1.1em;">% Reserva Total</h3>
                    <p style="margin:0; font-size: 2.2em; font-weight: bold;">{porc_reserva_total}</p>
                </div>
            </div>

            <h3 style="color:{PALETA_COLORES['primario']}; margin-top:30px;">Análisis Visual de la Distribución</h3>
            <div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: center; background: #fafafa; padding: 15px; border-radius: 8px;">
                <div style="flex: 2; min-width: 300px;">
                    <h4 style='text-align:center; color:{PALETA_COLORES['secundario']}; margin-bottom: 5px;'>Desglose por Modalidad</h4>
                    {grafico_barras_html}
                    <div style='text-align:center; font-size:0.9em; margin-top:10px;'>
                        <span style='display:inline-block; width:12px; height:12px; background:{PALETA_COLORES['general_ingreso']}; border-radius:3px;'></span> General Ingreso
                        <span style='display:inline-block; width:12px; height:12px; background:{PALETA_COLORES['reserva_ingreso']}; border-radius:3px; margin-left:10px;'></span> Reserva Ingreso
                        <br>
                        <span style='display:inline-block; width:12px; height:12px; background:{PALETA_COLORES['general_ascenso']}; border-radius:3px;'></span> General Ascenso
                        <span style='display:inline-block; width:12px; height:12px; background:{PALETA_COLORES['reserva_ascenso']}; border-radius:3px; margin-left:10px;'></span> Reserva Ascenso
                    </div>
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <h4 style='text-align:center; color:{PALETA_COLORES['secundario']}; margin-bottom: 5px;'>Distribución Global</h4>
                    {grafico_donut_html}
                </div>
            </div>

            <h3 style="color:{PALETA_COLORES['primario']}; margin-top:30px;">Resumen Numérico Detallado</h3>
            {self.generar_tabla_html()}

            <h3 style="color:{PALETA_COLORES['primario']}; margin-top:30px;">Notas y Advertencias del Cálculo</h3>
            <div style="background:#fff; border-left:5px solid {PALETA_COLORES['acento']}; padding:15px; border-radius:4px;">
                {self.generar_mensajes_html()}
            </div>

            <h3 style="color:{PALETA_COLORES['primario']}; margin-top:30px;">Conclusión y Hoja de Ruta</h3>
            <div style="background:#F5F5F5; padding:20px; border-radius:8px;">
                {self._generar_conclusion_base()}
            </div>
        </div>
        """

    def generar_pdf_en_memoria(self) -> Tuple[str, bytes]:
        pdf = PDF_Reporte('P', 'mm', 'Letter')
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.alias_nb_pages()
        pdf.add_page()

        fecha_generado = datetime.now(BOGOTA_TZ).strftime('%d/%m/%Y %H:%M:%S %Z') if BOGOTA_TZ else datetime.now().strftime('%d/%m/%Y %H:%M:%S')

        pdf.set_font('Helvetica', 'B', 18)
        pdf.cell(0, 10, 'Reporte de Simulación de Vacantes OPEC', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, self.nombre_entidad, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.set_font('Helvetica', '', 9)
        pdf.cell(0, 8, f"Generado: {fecha_generado}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)

        pdf.chapter_title('1. Parámetros de la Simulación')
        params_html = (
            f"- Total Vacantes OPEC: {self.total_opec}\n"
            f"- Opción de Cálculo: {self.datos_entrada.opcion_calculo_str}\n"
            f"- Vacantes para Ingreso: {self.datos_entrada.v_ingreso}\n"
            f"- Vacantes para Ascenso: {self.datos_entrada.v_ascenso}"
        )
        if self.datos_entrada.v_ascenso > 0:
            pcd_ascenso_str = 'Sí' if self.datos_entrada.hay_pcd_para_ascenso else 'No'
            params_html += f"\n- Existen servidores elegibles para ascenso?: {pcd_ascenso_str}"
        pdf.chapter_body_html(params_html)
        pdf.ln(5)

        pdf.chapter_title('2. Resultados Numéricos')
        pdf.add_pandas_table(self._preparar_datos_tabla().applymap(lambda x: x.replace('<strong>','').replace('</strong>','') if isinstance(x, str) else x))

        if self.total_opec > 0:
            if pdf.get_y() > 160: pdf.add_page()
            pdf.add_image_from_buffer(self.grafico_barras_buffer, "3. Distribución de Vacantes por Modalidad")
            if pdf.get_y() > 220: pdf.add_page()
            pdf.add_image_from_buffer(self.grafico_donut_buffer, "4. Distribución Global de la Reserva")

        if pdf.get_y() > 200: pdf.add_page()
        pdf.chapter_title('5. Notas y Advertencias del Cálculo')
        pdf.chapter_body_html(self.generar_mensajes_html())
        pdf.ln(5)

        pdf.chapter_title('6. Conclusión y Hoja de Ruta')
        pdf.chapter_body_html(self._generar_conclusion_base())

        filename = f"Reporte_OPEC_{''.join(c for c in self.nombre_entidad if c.isalnum())[:25]}_{datetime.now().strftime('%Y%m%d')}.pdf"
        try:
            return filename, bytes(pdf.output())
        except (FPDFException, Exception) as e:
            st.error(f"Ocurrió un error al generar el PDF: {e}")
            return "error.pdf", b""

# ==============================================================================
# INTERFAZ DE USUARIO CON STREAMLIT (DISEÑO MEJORADO)
# ==============================================================================
def aplicar_estilos_modernos():
    """ Inyecta CSS para un diseño más moderno y amigable. """
    st.markdown(f"""
    <style>
        /* --- Fuentes y Colores Base --- */
        html, body, [class*="st-"], .st-emotion-cache-1v0mbdj, .st-emotion-cache-1kyxreq, .st-emotion-cache-1y4p8pa, .st-emotion-cache-1p05sfz {{
            font-family: 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            background-color: {PALETA_COLORES['fondo_claro']};
        }}
        /* --- Contenedor Principal --- */
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }}
        /* --- Sidebar --- */
        .st-emotion-cache-10oheav {{
            background-color: {PALETA_COLORES['fondo_tarjeta']};
            border-right: 1px solid #ddd;
        }}
        .st-emotion-cache-10oheav h2, .st-emotion-cache-10oheav h3 {{
             color: {PALETA_COLORES['secundario']};
        }}
        /* --- Botones --- */
        .stButton>button {{
            border-radius: 8px;
            border: 2px solid {PALETA_COLORES['primario']};
            background-color: {PALETA_COLORES['primario']};
            color: {PALETA_COLORES['texto_claro']};
            padding: 12px 24px;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: {PALETA_COLORES['secundario']};
            border-color: {PALETA_COLORES['secundario']};
            color: {PALETA_COLORES['texto_claro']};
            transform: scale(1.02);
        }}
        /* --- Títulos y Texto --- */
        h1 {{ color: {PALETA_COLORES['secundario']}; }}
        h2, h3 {{ color: {PALETA_COLORES['primario']}; }}
        /* --- Tarjetas de Métricas --- */
        .st-emotion-cache-1k9g3sg {{
            background-color: {PALETA_COLORES['fondo_tarjeta']};
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid #eee;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
    </style>
    """, unsafe_allow_html=True)

def configurar_sidebar():
    """ Configura y muestra los controles de entrada en la barra lateral. """
    with st.sidebar:
        st.image("https://www.cnsc.gov.co/sites/default/files/2023-10/logo-cnsc.png", use_column_width=True)
        st.header("⚙️ Parámetros de Entrada")

        with st.expander("ℹ️ Guía Rápida de Uso", expanded=False):
            st.markdown("""
            **1. Datos Generales:** Ingresa el nombre de tu entidad y el número total de vacantes de la OPEC.
            **2. Distribución:** Elige cómo se reparten las vacantes entre Ingreso y Ascenso (70/30 por defecto).
            **3. Condición de Ascenso:** Responde si existen servidores de carrera con discapacidad que puedan optar a los cargos de ascenso. **Esta respuesta es crucial**.
            **4. Generar:** Haz clic en "Generar Simulación" para ver los resultados.
            """)

        st.subheader("1. Datos Generales")
        nombre_entidad = st.text_input(
            "Nombre Completo de la Entidad",
            placeholder="Ej: Alcaldía Mayor de Bogotá D.C.",
            help="El nombre que aparecerá en los reportes."
        )
        total_vacantes = st.number_input(
            "Total de Vacantes en la OPEC",
            min_value=0, value=100, step=1,
            help="Número total de empleos a proveer en el proceso de selección."
        )

        st.subheader("2. Distribución de Vacantes")
        distribucion_tipo = st.radio(
            "Método de Distribución",
            options=['Automático (70/30)', 'Manual'],
            horizontal=True,
            help="La normativa sugiere un 30% para Ascenso y 70% para Ingreso. Puedes ajustarlo manualmente si es necesario."
        )

        es_automatico = (distribucion_tipo == 'Automático (70/30)')
        default_ascenso = round(total_vacantes * 0.3)
        if es_automatico:
            vacantes_ascenso = default_ascenso
        else:
            vacantes_ascenso = st.number_input(
                "Vacantes para Ascenso (Manual)",
                min_value=0, max_value=total_vacantes, value=default_ascenso, step=1,
                help="Digite el número exacto de vacantes para la modalidad de Ascenso."
            )
        vacantes_ingreso = total_vacantes - vacantes_ascenso

        st.metric(label="Vacantes Ingreso", value=vacantes_ingreso)
        st.metric(label="Vacantes Ascenso", value=vacantes_ascenso)
        
        if vacantes_ingreso < 0:
            st.error("El número de vacantes de ascenso no puede superar el total.")

        st.subheader("3. Condición para Ascenso")
        pcd_para_ascenso = False
        if vacantes_ascenso > 0:
            respuesta_elegibilidad = st.radio(
                "¿Existen servidores de carrera con discapacidad que cumplan requisitos para los cargos de ascenso?",
                options=['Sí, existen servidores elegibles', 'No, no existen servidores elegibles'],
                index=1,
                help="Si la respuesta es 'No', la reserva de ascenso será 0 por disposición legal, independientemente del cálculo."
            )
            pcd_para_ascenso = (respuesta_elegibilidad.startswith('Sí'))
        else:
            st.info("No hay vacantes de ascenso, por lo que esta condición no aplica.", icon="ℹ️")

        return nombre_entidad, total_vacantes, distribucion_tipo, vacantes_ingreso, vacantes_ascenso, pcd_para_ascenso

def main():
    st.set_page_config(page_title="Simulador Reserva PcD - CNSC", page_icon="♿", layout="wide")
    aplicar_estilos_modernos()

    # --- TÍTULO PRINCIPAL ---
    st.title("Simulador de Reserva de Plazas para Personas con Discapacidad (PcD)")
    st.markdown("Herramienta de apoyo para la aplicación de la **Ley 2418 de 2024**, facilitando el cálculo de la reserva legal de empleos en la OPEC.")
    st.divider()

    # --- ENTRADAS EN EL SIDEBAR ---
    nombre_entidad, total_vacantes, distribucion_tipo, vacantes_ingreso, vacantes_ascenso, pcd_para_ascenso = configurar_sidebar()

    # --- ÁREA DE RESULTADOS ---
    placeholder_resultados = st.empty()
    with placeholder_resultados.container():
        st.info("⬅️ Por favor, complete los datos en el panel de la izquierda y haga clic en **'Generar Simulación'** para ver los resultados aquí.", icon="💡")

    if st.sidebar.button("🚀 Generar Simulación", use_container_width=True, type="primary"):
        if not nombre_entidad.strip():
            st.sidebar.error("⚠️ El nombre de la entidad es obligatorio.")
        elif vacantes_ingreso < 0:
            st.sidebar.error("⚠️ Error en la distribución. Revise las vacantes.")
        else:
            with st.spinner("⚙️ Procesando simulación... ¡Calculando futuro!"):
                datos_entrada = DatosEntrada(
                    total_opec=total_vacantes,
                    v_ingreso=vacantes_ingreso,
                    v_ascenso=vacantes_ascenso,
                    opcion_calculo_str=distribucion_tipo,
                    hay_pcd_para_ascenso=pcd_para_ascenso
                )
                resultados_sim = LogicaCalculo.determinar_resultados_finales(datos_entrada)
                reporte = GeneradorReporte(nombre_entidad.strip(), datos_entrada, resultados_sim)

            placeholder_resultados.empty() # Limpiar el mensaje de bienvenida
            with placeholder_resultados.container():
                st.success("¡Simulación completada con éxito!", icon="✅")
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
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Marco Normativo y Enlaces de Interés"):
            st.markdown("""
            - **Ley 2418 de 2024:** [Consulte la norma en Función Pública](https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=249256)
            - **Circular Externa CNSC:** [Vea la circular sobre el reporte de vacantes](https://www.cnsc.gov.co/sites/default/files/2025-02/circular-externa-2025rs011333-reportede-vacantes-definitivas-aplicacion-ley-2418-2024.pdf)
            - **Sitio Web CNSC:** [www.cnsc.gov.co](https://www.cnsc.gov.co)
            """)
    with col2:
        with st.expander("Acerca de este Simulador"):
            st.info(CREDITOS_SIMULADOR.replace("\n", "\n\n"), icon="©️")

if __name__ == '__main__':
    main()
