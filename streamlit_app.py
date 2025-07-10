 # ==============================================================================
# 1. CÓDIGO DEL SIMULADOR ADAPTADO PARA STREAMLIT
# Código original de: Edwin Arturo Ruiz Moreno - Comisionado Nacional del Servicio Civil
# Derechos Reservados CNSC © 2025
# Adaptación a Streamlit por: Asistente de IA de Google
# ==============================================================================

import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import textwrap
import io
import sys
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

# Se importa streamlit, el motor de la aplicación web
import streamlit as st

# Se importa pytz para manejar la zona horaria de Colombia
try:
    from pytz import timezone
    BOGOTA_TZ = timezone('America/Bogota')
except ImportError:
    print("⚠️ Advertencia: Librería 'pytz' no encontrada. La fecha del PDF usará la hora del servidor (UTC).")
    BOGOTA_TZ = None

# FPDF2 para la generación de PDF
from fpdf import FPDF, FPDFException
from fpdf.enums import XPos, YPos

# BeautifulSoup para limpiar HTML
BS4_AVAILABLE = False
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    pass

# --- Configuración Visual y Constantes ---
mpl.rcParams['figure.dpi'] = 120
PDF_DPI = 150
mpl.rcParams['font.family'] = 'sans-serif'
pd.set_option('display.float_format', lambda x: f'{x:.1f}')

PALETA_COLORES = {
    'ingreso_general': '#B2EBF2', 'ingreso_reserva': '#00838F', 'ascenso_general': '#FFECB3',
    'ascenso_reserva': '#FF8F00', 'texto_claro': '#FFFFFF', 'texto_oscuro': '#333333',
    'fondo_titulo': '#004D40', 'fondo_hover': '#E0F2F1', 'grid': '#CFD8DC',
    'mensaje_info': '#005f73', 'mensaje_aviso': '#ae2012', 'mensaje_ok': '#0a9396',
    'borde_claro': '#EEEEEE', 'primario': '#00796B', 'acento': '#FFC107'
}

CREDITOS_SIMULADOR = "Código original de: Edwin Arturo Ruiz Moreno - Comisionado Nacional del Servicio Civil\nDerechos Reservados CNSC © 2025"

# ==============================================================================
# 2. LÓGICA DE NEGOCIO Y CLASES (SIN CAMBIOS)
# Todas las clases y la lógica de cálculo se mantienen igual.
# ==============================================================================

class EstadoCalculo(Enum): NORMAL, CERO_VACANTES, AJUSTE_V1, AJUSTE_SIN_PCD = range(4)

@dataclass
class ModalidadResultados: total: int; reserva: int; general: int; estado: EstadoCalculo
@dataclass
class ResultadosSimulacion: ingreso: ModalidadResultados; ascenso: ModalidadResultados
@dataclass
class DatosEntrada: total_opec: int; opcion_calculo_str: str; hay_pcd_para_ascenso: bool; v_ingreso: int; v_ascenso: int

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#'); return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

class PDF_Reporte(FPDF):
    def __init__(self, orientation: str = 'P', unit: str = 'mm', format: str = 'A4'):
        super().__init__(orientation, unit, format)
        self.set_margins(20, 20, 20)
        self.active_font_family = 'Helvetica'
        self._configurar_fuentes()

    def _configurar_fuentes(self):
        font_paths = [ Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path(r"C:\Windows\Fonts\DejaVuSans.ttf"), Path("/System/Library/Fonts/Supplemental/DejaVuSans.ttf"), ]
        dejavu_path = next((path for path in font_paths if path.exists()), None)
        if dejavu_path:
            try:
                self.add_font('DejaVu', '', str(dejavu_path)); self.add_font('DejaVu', 'B', str(dejavu_path.with_name('DejaVuSans-Bold.ttf'))); self.add_font('DejaVu', 'I', str(dejavu_path.with_name('DejaVuSans-Oblique.ttf'))); self.add_font('DejaVu', 'BI', str(dejavu_path.with_name('DejaVuSans-BoldOblique.ttf'))); self.active_font_family = 'DejaVu'
            except (RuntimeError, FPDFException): st.sidebar.info("ℹ️ Nota: Fuentes DejaVu encontradas pero no se pudieron cargar. Se usará Helvetica.")
        else: st.sidebar.info("ℹ️ Nota: Fuentes DejaVu no encontradas para el PDF. Caracteres como '★' pueden no renderizarse.")

    def set_active_font(self, style: str = '', size: Optional[float] = None):
        self.set_font(self.active_font_family, style, size if size else self.font_size_pt)

    def footer(self):
        self.set_y(-15)
        self.set_active_font('I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}}', align='R')
        self.set_y(-18); self.set_x(self.l_margin); self.set_active_font('I', 7)
        for line in CREDITOS_SIMULADOR.replace("©", "(c)").split('\n'):
            self.cell(0, 4, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.set_text_color(0,0,0)

    def chapter_title(self, text: str):
        self.set_active_font('B', 11); self.set_text_color(*hex_to_rgb(PALETA_COLORES['primario']))
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.set_draw_color(*hex_to_rgb(PALETA_COLORES['acento'])); self.line(self.get_x(), self.get_y(), self.get_x()+40, self.get_y())
        self.ln(3)
        self.set_text_color(0,0,0)

    def chapter_body_html(self, html_content: str):
        self.set_active_font('', 9)
        text = html_content
        if BS4_AVAILABLE:
            processed_html = html_content.replace("</li>","\n").replace("</p>","\n").replace("<br>","\n").replace("</h4>","\n\n")
            text = BeautifulSoup(processed_html, "html.parser").get_text(separator=" ")
        else:
            import re; text = re.sub(r'<br\s*/?>', '\n', text); text = re.sub(r'</(p|li|h4)>', '\n', text); text = re.sub(r'<[^>]+>', '', text)
        for paragraph in text.split('\n'):
            cleaned_paragraph = " ".join(paragraph.strip().split())
            if not cleaned_paragraph: continue
            style = 'B' if "Pasos Siguientes y Consideraciones Clave:" in cleaned_paragraph else ''
            prefix = "• " if cleaned_paragraph.startswith("•") else ""
            if prefix: cleaned_paragraph = cleaned_paragraph.replace("•","").strip()
            self.set_active_font(style, 9)
            if prefix: self.cell(4, 5, prefix); self.multi_cell(0, 5, cleaned_paragraph, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
            else: self.multi_cell(0, 5, cleaned_paragraph, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
            self.ln(2)

    def add_pandas_table(self, df: pd.DataFrame):
        self.set_active_font('B', 8); available_width = self.w - self.l_margin - self.r_margin
        col_widths = [available_width * w for w in [0.25, 0.22, 0.28, 0.25]]; line_height = 7
        self.set_fill_color(*hex_to_rgb(PALETA_COLORES['fondo_titulo'])); self.set_text_color(*hex_to_rgb(PALETA_COLORES['texto_claro']))
        for i, header in enumerate(df.columns): self.cell(col_widths[i], line_height, header, border=0, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C', fill=True)
        self.ln(line_height); self.set_active_font('', 8); self.set_text_color(0,0,0)
        for index, row in df.iterrows():
            is_total_row = 'TOTAL' in row['Modalidad']
            if is_total_row: self.set_active_font('B', 8.5)
            fill_color = hex_to_rgb(PALETA_COLORES['fondo_hover']) if is_total_row else (255,255,255)
            self.set_fill_color(*fill_color)
            for i, datum in enumerate(row):
                align = 'L' if i == 0 else 'C'
                self.cell(col_widths[i], line_height, str(datum), border='T', new_x=XPos.RIGHT, new_y=YPos.TOP, align=align, fill=True)
            self.ln(line_height)
            if is_total_row: self.set_active_font('', 8)
        self.ln(4)

    def add_image_from_buffer(self, img_buffer: io.BytesIO, title: str):
        self.set_active_font('B', 11); self.set_text_color(*hex_to_rgb(PALETA_COLORES['primario']))
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L'); self.ln(2)
        self.set_text_color(0,0,0)
        img_buffer.seek(0); self.image(img_buffer, w=self.w - self.l_margin - self.r_margin)
        plt.close('all'); self.ln(5)

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
        r_asc, e_asc = (0, EstadoCalculo.AJUSTE_SIN_PCD) if v_ascenso > 0 and not datos_entrada.hay_pcd_para_ascenso else LogicaCalculo._calcular_reserva_individual(v_ascenso)
        return ResultadosSimulacion(ingreso=ModalidadResultados(total=v_ingreso, reserva=r_ing, general=max(0, v_ingreso - r_ing), estado=e_ing), ascenso=ModalidadResultados(total=v_ascenso, reserva=r_asc, general=max(0, v_ascenso - r_asc), estado=e_asc))

class GeneradorReporte:
    def __init__(self, nombre_entidad: str, datos_entrada: DatosEntrada, resultados: ResultadosSimulacion):
        self.nombre_entidad = nombre_entidad; self.datos_entrada = datos_entrada; self.resultados = resultados
        self.total_opec = datos_entrada.total_opec; self.escala_pictograma = 1
        self.pdf_font_family = 'DejaVu' if any(p.exists() for p in [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]) else 'Helvetica'
        self.graficos_img_buffer = self._generar_todos_los_graficos()

    def _calcular_porcentaje_str(self, valor: float, total: float) -> str: return "0.0%" if total <= 0 else f"{(valor / total) * 100:.1f}%"
    def _preparar_datos_tabla(self) -> pd.DataFrame:
        i, a = self.resultados.ingreso, self.resultados.ascenso
        tr, tg = i.reserva + a.reserva, i.general + a.general
        reserva_ing_str = f"{i.reserva} ({self._calcular_porcentaje_str(i.reserva, i.total)})" if i.total > 0 else f"{i.reserva}"
        reserva_asc_str = f"{a.reserva} ({self._calcular_porcentaje_str(a.reserva, a.total)})" if a.total > 0 else f"{a.reserva}"
        return pd.DataFrame({'Modalidad': ['Ingreso','Ascenso','TOTAL'],'Vacantes Totales': [i.total, a.total, self.total_opec],'Reserva PcD': [reserva_ing_str, reserva_asc_str, tr],'Concurso General': [i.general, a.general, tg]})
    def generar_tabla_html(self) -> str:
        df_styled = self._preparar_datos_tabla().astype(str)
        df_styled.iloc[-1] = df_styled.iloc[-1].apply(lambda x: f"<strong>{x}</strong>")
        styles=[dict(selector="th", props=[("font-size","11pt"),("text-align","center"),("background-color",PALETA_COLORES['fondo_titulo']),("color",PALETA_COLORES['texto_claro'])]), dict(selector="td", props=[("font-size","10.5pt"),("text-align","center"),("border",f"1px solid {PALETA_COLORES['borde_claro']}"),("color",PALETA_COLORES['texto_oscuro']),("background-color","#FFFFFF")])]
        return df_styled.style.set_table_styles(styles).hide(axis="index").to_html(escape=False)
    def _generar_mensajes_base(self) -> List[str]:
        mensajes = []
        r = self.resultados
        if r.ascenso.estado == EstadoCalculo.AJUSTE_SIN_PCD: mensajes.append(f"<li><strong>Ajuste en Ascenso:</strong> Se indicó que no existen servidores que cumplan los requisitos (derechos de carrera y discapacidad certificada) para la modalidad de ascenso, por lo que la reserva se ajusta a <strong>0</strong>.</li>")
        for m_name, m_data in [('INGRESO', r.ingreso), ('ASCENSO', r.ascenso)]:
            if m_data.estado == EstadoCalculo.AJUSTE_V1: mensajes.append(f"<li><strong>Nota ({m_name}):</strong> Con solo <strong>1 vacante</strong>, no se aplica reserva.</li>")
        if r.ascenso.reserva > 0:
            mensajes.append(f"<li><strong>Nota Importante sobre Ascenso:</strong> La reserva de <strong>{r.ascenso.reserva}</strong> vacante(s) está condicionada a la existencia de servidores que posean <strong>derechos de carrera administrativa</strong> y, a su vez, tengan una <strong>discapacidad debidamente certificada</strong> y cumplan los demás requisitos. De ser así, la reserva es obligatoria y deberá constituirse con empleos que garanticen el ascenso, es decir, la <strong>movilidad vertical ascendente</strong> dentro de la planta de personal.</li>")
        if not mensajes and self.total_opec > 0: mensajes.append(f"<li>No se generaron advertencias especiales.</li>")
        elif self.total_opec == 0: mensajes.append(f"<li>No hay vacantes para calcular.</li>")
        return mensajes
    def generar_mensajes_html(self) -> str: return f"<ul style='padding-left:20px;font-size:0.95em;color:{PALETA_COLORES['texto_oscuro']};'>{''.join(self._generar_mensajes_base())}</ul>"
    def _generar_conclusion_base(self) -> str:
        pasos_siguientes = """<h4 style='margin-top:15px; margin-bottom:5px; color:#004D40;'>Pasos Siguientes y Consideraciones Clave:</h4><ul style='padding-left:20px;font-size:0.9em; line-height:1.6;'><li><strong>Representatividad Jerárquica:</strong> Se debe procurar que la reserva de empleos refleje la diversidad de los niveles jerárquicos de la entidad (Directivo, Asesor, Profesional, Técnico y Asistencial), evitando una concentración exclusiva en los niveles de menor rango.</li><li><strong>Análisis de Empleos:</strong> Las vacantes seleccionadas para la reserva deben ser objeto de un estudio que incluya el análisis de funciones, las características del puesto de trabajo y los ajustes razonables que podrían implementarse.</li><li><strong>Uso del "Recomendador de Empleos PcD":</strong> Se invita a la entidad a usar la herramienta complementaria <strong>"Recomendador de empleos para la reserva"</strong> de la CNSC. Para acceder, contactar a <strong>Edwin Arturo Ruiz Moreno</strong> de la CNSC.</li><li><strong>Validación Profesional:</strong> Los resultados deben ser analizados y validados por un profesional en Salud y Seguridad en el Trabajo (SST) o por la Administradora de Riesgos Laborales (ARL) correspondiente.</li></ul>"""
        return (pasos_siguientes if self.total_opec > 0 else "")
    def _render_fig_to_buffer(self, fig: plt.Figure) -> io.BytesIO:
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=PDF_DPI, bbox_inches='tight', transparent=True); buf.seek(0); plt.close(fig); return buf
    def _crear_grafico_distribucion_unificado(self) -> Optional[io.BytesIO]:
        if self.total_opec == 0: return None
        r = self.resultados; fig, axs = plt.subplots(1, 2, figsize=(9, 4.5))
        colors_ingreso = [PALETA_COLORES['ingreso_general'], PALETA_COLORES['ingreso_reserva']]; colors_ascenso = [PALETA_COLORES['ascenso_general'], PALETA_COLORES['ascenso_reserva']]
        def label_formatter(pct, allvals):
            absolute = int(round(pct/100.*sum(allvals))); return f"{absolute}\n({pct:.1f}%)" if pct > 3 else ''
        all_data = [(r.ingreso.general, r.ingreso.reserva), (r.ascenso.general, r.ascenso.reserva)]
        for i, (ax, data, colors, title) in enumerate(zip(axs, all_data, [colors_ingreso, colors_ascenso], ["Modalidad Ingreso", "Modalidad Ascenso"])):
            total_modalidad = sum(data)
            if total_modalidad > 0:
                wedges, texts, autotexts = ax.pie(data,colors=colors,autopct=lambda p: label_formatter(p, data),startangle=90,counterclock=False,wedgeprops=dict(width=0.45, edgecolor='w'),pctdistance=0.78,textprops={'color':"w", 'fontsize':8, 'fontweight':'bold'})
                autotexts[0].set_color(PALETA_COLORES['texto_oscuro']); autotexts[1].set_color(PALETA_COLORES['texto_claro'])
                ax.text(0, 0, f"{total_modalidad}\nVacantes", ha='center', va='center', fontsize=16, fontweight='bold', color=PALETA_COLORES['texto_oscuro'])
            else:
                ax.pie([1], colors=['#f0f0f0'], wedgeprops=dict(width=0.45)); ax.text(0, 0, f"0\nVacantes", ha='center', va='center', fontsize=16, color='#bbbbbb')
            ax.set_title(title, fontsize=12, pad=10, color=PALETA_COLORES['primario'])
        plt.tight_layout(rect=[0, 0, 1, 0.95]); return self._render_fig_to_buffer(fig)
    def _crear_pictograma(self) -> Optional[io.BytesIO]:
        if self.total_opec == 0: return None
        res, max_ico = self.resultados, 150; s_res = "★" if self.pdf_font_family == 'DejaVu' else "*"; s_gen = "●"
        self.escala_pictograma = 1 if self.total_opec <= max_ico else math.ceil(self.total_opec / max_ico)
        iconos = ([{'s':s_res,'c':PALETA_COLORES['ingreso_reserva']}]*round(res.ingreso.reserva/self.escala_pictograma) + [{'s':s_gen,'c':PALETA_COLORES['ingreso_general']}]*round(res.ingreso.general/self.escala_pictograma) + [{'s':s_res,'c':PALETA_COLORES['ascenso_reserva']}]*round(res.ascenso.reserva/self.escala_pictograma) + [{'s':s_gen,'c':PALETA_COLORES['ascenso_general']}]*round(res.ascenso.general/self.escala_pictograma))
        if not iconos: return None
        cols = 25; rows = math.ceil(len(iconos) / cols)
        fig, ax = plt.subplots(figsize=(10, rows * 0.45)); ax.set_xlim(-0.5, cols - 0.5); ax.set_ylim(-0.5, rows - 0.5); ax.axis("off"); ax.invert_yaxis()
        font_for_plot = 'DejaVu Sans' if self.pdf_font_family == 'DejaVu' else 'sans-serif'
        for i, icon in enumerate(iconos): ax.text(i % cols, i // cols, icon['s'], color=icon['c'], fontsize=12, ha='center',va='center',fontfamily=font_for_plot)
        plt.tight_layout(); return self._render_fig_to_buffer(fig)
    def _generar_pictograma_explicacion(self, para_pdf=False) -> str:
        s_res = ("★" if self.pdf_font_family == 'DejaVu' else "*") if para_pdf else "★"
        res = self.resultados
        items = [(res.ingreso.reserva, f"{s_res} Reserva Ingreso ({res.ingreso.reserva})"), (res.ingreso.general, f"● General Ingreso ({res.ingreso.general})"), (res.ascenso.reserva, f"{s_res} Reserva Ascenso ({res.ascenso.reserva})"), (res.ascenso.general, f"● General Ascenso ({res.ascenso.general})")]
        texto_items = "  |  ".join(text for count, text in items if count > 0)
        nota_escala = f" | Nota: Cada símbolo representa aprox. {self.escala_pictograma} vacantes." if self.escala_pictograma > 1 else ""
        return f"Convenciones: {texto_items}{nota_escala}"
    def _generar_todos_los_graficos(self) -> Dict[str, Optional[io.BytesIO]]:
        return {"unificado": self._crear_grafico_distribucion_unificado(), "pictograma": self._crear_pictograma()}
    def generar_reporte_html_completo(self):
        from base64 import b64encode
        def img(b: Optional[io.BytesIO]) -> str: return f'<img src="data:image/png;base64,{b64encode(b.getvalue()).decode("utf-8")}" style="width:100%;max-width:700px;margin:auto;display:block;"/>' if b else ""
        pictograma_html, explicacion_html = img(self.graficos_img_buffer.get('pictograma')), self._generar_pictograma_explicacion()
        html_string = f"""<div style="font-family:sans-serif;border:1px solid #ddd;border-radius:8px;padding:20px;background:#f9f9f9;color:{PALETA_COLORES['texto_oscuro']};"><h1 style="color:{PALETA_COLORES['fondo_titulo']};border-bottom:2px solid {PALETA_COLORES['acento']};padding-bottom:10px;">📊 Reporte de Simulación: {self.nombre_entidad}</h1><div>{img(self.graficos_img_buffer.get('unificado'))}</div><h2 style="color:{PALETA_COLORES['primario']};margin-top:25px;">Resumen de Distribución</h2>{self.generar_tabla_html()}<h2 style="color:{PALETA_COLORES['primario']};margin-top:25px;">Notas y Advertencias Clave</h2><div style="background:#fff;border-left:5px solid {PALETA_COLORES['acento']};padding:1px 15px;border-radius:4px;">{self.generar_mensajes_html()}</div><h2 style="color:{PALETA_COLORES['primario']};margin-top:25px;">Pictograma Visual</h2><div style="background:#fff;padding:15px;border-radius:4px;">{pictograma_html}<p style="text-align:center;font-size:0.9em;color:#666;margin-top:10px;">{explicacion_html}</p></div><h2 style="color:{PALETA_COLORES['primario']};margin-top:25px;">Conclusión y Pasos Siguientes</h2><div style="background:{PALETA_COLORES['fondo_hover']};padding:15px;border-radius:4px;">{self._generar_conclusion_base()}</div></div>"""
        st.markdown(html_string, unsafe_allow_html=True)
    def generar_pdf_en_memoria(self) -> Tuple[str, bytes]:
        pdf = PDF_Reporte(); pdf.set_auto_page_break(auto=True, margin=20); pdf.alias_nb_pages(); pdf.add_page()
        fecha_generado = datetime.now(BOGOTA_TZ).strftime('%d/%m/%Y %H:%M:%S %Z') if BOGOTA_TZ else datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        pdf.set_active_font('B', 16); pdf.cell(0, 10, 'Reporte de Simulación de Vacantes OPEC', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.set_active_font('', 12); pdf.cell(0, 8, self.nombre_entidad, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.set_active_font('', 9); pdf.cell(0, 6, f"Generado: {fecha_generado}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C'); pdf.ln(8)
        pdf.chapter_title('Parámetros de la Simulación')
        params_html = f"• Total Vacantes OPEC: {self.total_opec}\n• Opción de Cálculo: {self.datos_entrada.opcion_calculo_str}\n• Vacantes Ingreso: {self.datos_entrada.v_ingreso}\n• Vacantes Ascenso: {self.datos_entrada.v_ascenso}"
        if self.datos_entrada.v_ascenso > 0: params_html += f"\n• ¿Existen servidores elegibles para ascenso?: {'Sí' if self.datos_entrada.hay_pcd_para_ascenso else 'No'}"
        pdf.chapter_body_html(params_html)
        pdf.chapter_title('Resultados Numéricos'); pdf.add_pandas_table(self._preparar_datos_tabla())
        if buffer := self.graficos_img_buffer.get("unificado"): pdf.add_image_from_buffer(buffer, "Distribución General de Vacantes")
        if pdf.get_y() > 180: pdf.add_page() # Evitar que el contenido se corte
        pdf.chapter_title('Notas y Advertencias del Cálculo'); pdf.chapter_body_html(self.generar_mensajes_html())
        if self.total_opec > 0:
            pdf.chapter_title("Desglose Visual de la Oferta")
            if buffer := self.graficos_img_buffer.get("pictograma"):
                pdf.image(buffer, w=pdf.w - pdf.l_margin - pdf.r_margin)
                pdf.set_active_font('I', 8); pdf.multi_cell(0, 5, self._generar_pictograma_explicacion(para_pdf=True), align='C'); pdf.ln(5)
        pdf.chapter_title('Conclusión y Pasos Siguientes'); pdf.chapter_body_html(self._generar_conclusion_base())
        filename = f"Reporte_OPEC_{''.join(c for c in self.nombre_entidad if c.isalnum())[:30]}_{datetime.now(BOGOTA_TZ).strftime('%Y%m%d') if BOGOTA_TZ else datetime.now().strftime('%Y%m%d')}.pdf"
        try:
            pdf_output = pdf.output(dest='S').encode('latin-1')
            return filename, pdf_output
        except (FPDFException, Exception) as e:
            st.error(f"Ocurrió un error al generar el PDF: {e}")
            return "error.pdf", b""

# ==============================================================================
# 3. INTERFAZ DE USUARIO CON STREAMLIT
# Esta sección reemplaza la clase SimuladorUI de ipywidgets.
# ==============================================================================
def main():
    st.set_page_config(page_title="Simulador OPEC", page_icon="📊", layout="wide")

    st.title("✨ Simulador Interactivo de Vacantes OPEC")
    st.markdown("Complete los campos para calcular la distribución de vacantes y la reserva para Personas con Discapacidad (PcD).")
    st.markdown("---")
    
    st.sidebar.title("Acerca de")
    st.sidebar.info(CREDITOS_SIMULADOR.replace("\n", "\n\n"))

    # Formulario para agrupar las entradas
    with st.form(key="simulador_form"):
        st.header("1. Datos Generales")
        nombre_entidad = st.text_input("Nombre de la Entidad:", placeholder="Ej: Alcaldía Mayor de Bogotá D.C.")
        total_vacantes = st.number_input("Total de Vacantes en la OPEC:", min_value=0, value=100, step=1)

        st.header("2. Distribución de Vacantes")
        distribucion_tipo = st.radio(
            "Seleccione el método de distribución:",
            options=[('Automático (70% Ingreso / 30% Ascenso)', 'auto'), ('Manual', 'manual')],
            horizontal=True, key="dist_tipo"
        )
        
        ascenso_manual = 0
        if distribucion_tipo == 'manual':
            ascenso_manual = st.number_input("Número de Vacantes para Ascenso:", min_value=0, max_value=total_vacantes, value=30, step=1)
        
        # Calcular vacantes de ascenso si es automático
        vacantes_ascenso = ascenso_manual if distribucion_tipo == 'manual' else round(total_vacantes * 0.3)
        vacantes_ingreso = total_vacantes - vacantes_ascenso

        st.info(f"Cálculo actual: **{vacantes_ingreso}** para Ingreso y **{vacantes_ascenso}** para Ascenso.")

        st.header("3. Condición para Ascenso")
        pcd_para_ascenso = True
        if vacantes_ascenso > 0:
             pcd_para_ascenso = st.radio(
                "¿Existen servidores con derechos de carrera Y discapacidad certificada que cumplen los requisitos para el ascenso?",
                options=[('Sí', True), ('No', False)],
                horizontal=True, key="pcd_ascenso"
             )
        else:
            st.write("No hay vacantes de ascenso para evaluar esta condición.")

        # Botón para enviar el formulario
        submit_button = st.form_submit_button(label="✨ Generar Simulación", use_container_width=True)

    # Lógica que se ejecuta después de presionar el botón
    if submit_button:
        # --- Validación de Entradas ---
        if not nombre_entidad.strip():
            st.error("⚠️ **Error:** El nombre de la entidad es obligatorio.")
        elif vacantes_ingreso < 0:
            st.error("⚠️ **Error:** El número de vacantes de ingreso no puede ser negativo. Ajuste la distribución manual.")
        else:
            with st.spinner("⚙️ Procesando, por favor espere..."):
                time.sleep(1) # Pequeña pausa para mejorar la UX
                
                # --- Ejecución del Cálculo ---
                opcion_str = 'Automático (70/30)' if distribucion_tipo == 'auto' else 'Manual'
                datos_entrada = DatosEntrada(
                    total_opec=total_vacantes,
                    v_ingreso=vacantes_ingreso,
                    v_ascenso=vacantes_ascenso,
                    opcion_calculo_str=opcion_str,
                    hay_pcd_para_ascenso=pcd_para_ascenso if vacantes_ascenso > 0 else False
                )
                
                resultados_sim = LogicaCalculo.determinar_resultados_finales(datos_entrada)
                reporte = GeneradorReporte(nombre_entidad.strip(), datos_entrada, resultados_sim)

                # --- Generación de Salidas ---
                st.success("¡Simulación completada! ✨")
                reporte.generar_reporte_html_completo()
                
                pdf_filename, pdf_bytes = reporte.generar_pdf_en_memoria()
                
                if pdf_bytes:
                    st.download_button(
                        label="📄 Descargar Reporte en PDF",
                        data=pdf_bytes,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        use_container_width=True
                    )

if __name__ == '__main__':
    main()
