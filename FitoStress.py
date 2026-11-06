#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iPASTIC Desktop v2.5
Single-file application for calculating plant stress tolerance indices,
performing cluster analysis, and visualizing results.
"""

import json
import sys
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Import PyQt5 components
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
                             QMenuBar, QToolBar, QAction, QMessageBox, QFileDialog, QInputDialog, QDockWidget,
                             QStatusBar, QComboBox, QPushButton, QLabel, QLineEdit, QFormLayout, QHeaderView,
                             QStyleFactory, QStyle, QColorDialog, QMenu, QShortcut, QKeySequenceEdit, QHBoxLayout, QDialog,
                             QProgressBar, QSpinBox, QSplitter, QGroupBox, QCheckBox, QTextEdit, QListWidget, QScrollArea)
from PyQt5.QtGui import QIcon, QPixmap, QFont, QColor, QKeySequence, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, pyqtSlot, QSize, QTimer

# Core scientific computing
import warnings
warnings.filterwarnings('ignore')

# Import enhanced analyses
try:
    from enhanced_analyses import EnhancedAnalyses
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    logging.warning("Enhanced analyses not available. Some advanced features will be disabled.")

# Plotting - defer backend selection to avoid Qt widget creation before QApplication
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Statistical analysis
import scipy.stats as stats
from scipy.cluster.hierarchy import dendrogram, linkage
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera

SUPPORTED_LANGUAGES = {
    'PTBR': 'PTBR.json',
    'EN': 'EN.json',
}
Language_config_file = os.path.join(os.path.dirname(__file__), 'language_config.json')
try:
    with open(Language_config_file, 'r', encoding='utf-8') as f:        
        lang_config = json.load(f)
        global LANGUAGE_CODE 
        LANGUAGE_CODE = lang_config.get("LanguageCode")
except Exception as e:
    LANGUAGE_CODE = 'PTBR'
LANG_DATA = {}


def load_language(lang_code):
    global LANGUAGE_CODE, LANG_DATA
    if lang_code not in SUPPORTED_LANGUAGES:
        lang_code = 'PTBR'
    LANGUAGE_CODE = lang_code
    language_file = os.path.join(os.path.dirname(__file__), SUPPORTED_LANGUAGES[lang_code])
    try:
        with open(language_file, encoding='utf-8') as f:
            LANG_DATA = json.load(f)
    except Exception as e:
        logging.warning(f"Could not load language file '{language_file}': {e}")
        LANG_DATA = {}


def GetFromLanguage(source, index):
    try:
        return LANG_DATA.get(source, [])[index]
    except Exception:
        return f"{source}.{index}"


load_language(LANGUAGE_CODE)

class MainWindow(QMainWindow):
    """
    The main application window, containing the UI and event handling logic.
    """
    def __init__(self):
        try:
            super().__init__()
            
            # Initialize settings (after QApplication exists)  
            self.settings_handler = None
            self.classification_limits = {'tolerant': 20.0, 'mod_tolerant': 40.0, 'mod_susceptible': 60.0}

            self.setWindowTitle(GetFromLanguage("MainWindowTitle", 0))
            # self.setGeometry(100, 100, 1300, 850)  # Comentado para permitir maximização
            try:
                self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
            except Exception:
                pass

            # Initialize data holders and analysis objects
            self.data = pd.DataFrame()
            self.indices = pd.DataFrame()
            self.rankings_df = pd.DataFrame()
            self.original_column_names = [GetFromLanguage("Headers", i) for i in range(3)]
            self.current_graph_fig = None
            self.current_canvas = None
            self.last_click_time = 0
            try:
                self.analysis = IpasticAnalysis()
                self.enhanced_analysis = EnhancedAnalyses() if ENHANCED_AVAILABLE else None
            except Exception as e:
                QMessageBox.critical(self, "Erro ao inicializar análise", f"Erro: {str(e)}")
                self.analysis = None
                self.enhanced_analysis = None
            try:
                self.data_manager = DataManager()
            except Exception as e:
                QMessageBox.critical(self, "Erro ao inicializar DataManager", f"Erro: {str(e)}")
                self.data_manager = None
            
            # Initialize report generator
            self.report_generator = ReportGenerator(self)
            self.worker = None

            # Add a timer for debouncing calculations
            self.calculation_timer = QTimer(self)
            self.calculation_timer.setSingleShot(True)
            self.calculation_timer.timeout.connect(self.start_calculation)

            self.init_graph_controls()
            self.init_ui()
            self.init_settings() # Initialize settings after UI is ready
            self.update_ui_state() # Set initial state of widgets
            self.setAcceptDrops(True) # Enable drag and drop
        except Exception as e:
            QMessageBox.critical(None, "Erro na Inicialização", f"Erro ao inicializar a janela principal:\n{str(e)}")

    def init_settings(self):
        """Initialize settings after QApplication is available."""
        try:
            self.settings_handler = QSettings("Plant Stress Lab", "iPASTIC")
            # Load settings if available
            if self.settings_handler:
                saved_limits = self.settings_handler.value("classification_limits", None)
                if saved_limits:
                    self.classification_limits = saved_limits
        except Exception as e:
            print(f"Warning: Could not initialize settings: {e}")
            self.settings_handler = None

    def init_ui(self):
        """Initializes all UI components."""
        self.create_actions()
        self.create_menu()
        self.create_toolbar()

        self.central_widget = QTabWidget()
        self.setCentralWidget(self.central_widget)

        # Initialize advanced config widgets before creating tabs that use them
        self.pca_components_spin = QSpinBox()
        self.pca_components_spin.setRange(2, 10)
        self.pca_components_spin.setValue(2)
        
        # Novos controles PCA avançados
        self.pca_component_x_combo = QComboBox()
        self.pca_component_y_combo = QComboBox()
        self.pca_matrix_combo = QComboBox()
        self.pca_matrix_combo.addItems(['Correlação', 'Covariância'])
        self.pca_matrix_combo.setCurrentText('Correlação')
        self.pca_show_variance_check = QCheckBox("Mostrar percentuais")
        self.pca_show_variance_check.setChecked(True)
        
        self.pca_show_vectors_check = QCheckBox("Mostrar vetores de carregamento")
        self.pca_show_vectors_check.setChecked(True)
        
        # Inicializar combos de componentes
        for i in range(1, 11):  # PC1 até PC10
            self.pca_component_x_combo.addItem(f"PC{i}")
            self.pca_component_y_combo.addItem(f"PC{i}")
        
        # Definir valores padrão
        self.pca_component_x_combo.setCurrentText("PC1")
        self.pca_component_y_combo.setCurrentText("PC2")
        self.dendro_method_combo = QComboBox()
        self.dendro_method_combo.addItems(['ward', 'single', 'complete', 'average'])
        self.dendro_metric_combo = QComboBox()
        self.dendro_metric_combo.addItems(['euclidean', 'cityblock', 'cosine', 'correlation'])
        
        # Controles para divisão em grupos
        self.dendro_groups_spin = QSpinBox()
        self.dendro_groups_spin.setMinimum(1)
        self.dendro_groups_spin.setMaximum(10)
        self.dendro_groups_spin.setValue(1)  # Padrão: 1 grupo (tudo preto) 
        
        self.dendro_color_groups_check = QCheckBox("Colorir grupos")
        self.dendro_color_groups_check.setChecked(False)  # Padrão: desabilitado (preto)
        self.heatmap_method_combo = QComboBox()
        self.heatmap_method_combo.addItems(['pearson', 'spearman', 'kendall'])
        self.heatmap_values_check = QCheckBox("Mostrar valores numéricos")
        self.heatmap_values_check.setChecked(True)

        # Create tabs
        self.tab_input = self.create_input_tab()
        self.tab_indices = self.create_indices_tab()
        self.tab_rankings = self.create_rankings_tab()
        self.tab_statistics = self.create_statistics_tab()
        self.tab_graphics = self.create_graphics_tab()
        self.tab_regression = self.create_regression_tab()
        self.tab_reports = self.create_reports_tab()
        self.tab_help = self.create_help_tab()

        self.central_widget.addTab(self.tab_input, "1. Entrada de Dados")
        self.central_widget.addTab(self.tab_indices, GetFromLanguage("Tabs", 1))
        self.central_widget.addTab(self.tab_rankings, GetFromLanguage("Tabs", 2))
        self.central_widget.addTab(self.tab_statistics, GetFromLanguage("Tabs", 3))
        self.central_widget.addTab(self.tab_graphics, GetFromLanguage("Tabs", 4))
        self.central_widget.addTab(self.tab_pca_results, GetFromLanguage("Tabs", 5))
        self.central_widget.addTab(self.tab_regression, GetFromLanguage("Tabs", 6))
        self.central_widget.addTab(self.tab_reports, GetFromLanguage("Tabs", 7))
        self.central_widget.addTab(self.tab_help, GetFromLanguage("Tabs", 8))

        # Add shortcuts for copy and paste
        QShortcut(QKeySequence.Copy, self.table_input, lambda: self.copy_table_selection(self.table_input))
        QShortcut(QKeySequence.Paste, self.table_input, self.paste_into_input_table)
        QShortcut(QKeySequence.Copy, self.table_indices, lambda: self.copy_table_selection(self.table_indices))
        QShortcut(QKeySequence.Copy, self.table_rankings, lambda: self.copy_table_selection(self.table_rankings))

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def on_table_item_changed(self, item):
        """
        Called when a cell in the input table is changed by the user.
        Reads the current data from the table, updates the internal DataFrame,
        and refreshes the UI state.
        """
        # Disconnect the signal to prevent recursive calls while we process the table
        try:
            self.table_input.itemChanged.disconnect(self.on_table_item_changed)
        except TypeError:
            pass  # Signal might have already been disconnected

        rows = self.table_input.rowCount()
        cols = self.table_input.columnCount()
        
        # Use original column names for the DataFrame
        headers = self.original_column_names if len(self.original_column_names) == cols else [f'Col{i+1}' for i in range(cols)]

        table_data = []
        for r in range(rows):
            row_items = [self.table_input.item(r, c) for c in range(cols)]
            # Only include rows that have at least one non-empty cell
            if any(item and item.text().strip() for item in row_items):
                table_data.append([item.text().strip() if item else '' for item in row_items])

        if not table_data:
            self.data = pd.DataFrame()
        else:
            # Create a DataFrame from the table content
            df = pd.DataFrame(table_data, columns=headers)
            
            # Basic validation and standardization similar to file import
            try:
                if df.shape[1] == 2:
                    df.columns = ['Yp', 'Ys']
                    df['Genotipo'] = [f'Genótipo_{i+1}' for i in range(len(df))]
                    df = df[['Genotipo', 'Yp', 'Ys']]
                else:
                    df.columns = ['Genotipo', 'Yp', 'Ys']

                # Coerce to numeric, but don't raise errors, just for the state check
                df['Yp'] = pd.to_numeric(df['Yp'], errors='coerce')
                df['Ys'] = pd.to_numeric(df['Ys'], errors='coerce')
                
                # Drop rows where essential data is missing after coercion
                df.dropna(subset=['Yp', 'Ys'], how='all', inplace=True)
                self.data = df
            except (ValueError, TypeError) as e:
                # In case of error (e.g., wrong column count), treat as no data
                self.data = pd.DataFrame()
                logging.warning(f"Could not parse table data for UI update: {e}")

        # Reconnect the signal for future edits
        self.table_input.itemChanged.connect(self.on_table_item_changed)
        
        # Update the UI based on whether we have valid data now
        self.update_ui_state()

        # Automatically start calculation if data is valid (with debounce)
        if self.calc_action.isEnabled():
            self.calculation_timer.start(500) # Debounce for 500ms

    def create_actions(self):
        """Creates QAction objects for menus and toolbars."""
        self.import_action = QAction(self.style().standardIcon(QStyle.SP_DirIcon), GetFromLanguage("MenuFile", 1), self,
                                     shortcut=QKeySequence.Open, triggered=self.import_data)
        self.export_excel_action = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), GetFromLanguage("MenuFile", 2), self,
                                           shortcut="Ctrl+E", triggered=self.export_to_excel)
        self.clear_action = QAction(self.style().standardIcon(QStyle.SP_TrashIcon), GetFromLanguage("MenuFile", 3), self,
                                    shortcut="Ctrl+D", triggered=self.clear_all_data)
        self.exit_action = QAction(GetFromLanguage("MenuFile", 4), self, shortcut="Ctrl+Q", triggered=self.close)

        self.change_language = QAction(self.style().standardIcon(QStyle.SP_DesktopIcon), GetFromLanguage("MenuFile", 5), self,
                                       triggered=self.change_language)
        
        self.calc_action = QAction(self.style().standardIcon(QStyle.SP_MediaPlay), GetFromLanguage("MenuTools", 1), self,
                                   triggered=self.start_calculation)
        self.settings_action = QAction(self.style().standardIcon(QStyle.SP_FileDialogDetailedView), GetFromLanguage("MenuTools", 2), self,
                                       triggered=self.open_settings)
        
        # Report actions
        self.indices_report_action = QAction(self.style().standardIcon(QStyle.SP_FileDialogDetailedView), GetFromLanguage("MenuReports", 1), self,
                                            triggered=self.generate_indices_report)
        self.statistics_report_action = QAction(self.style().standardIcon(QStyle.SP_FileDialogDetailedView), GetFromLanguage("MenuReports", 2), self,
                                               triggered=self.generate_statistics_report)
        self.clustering_report_action = QAction(self.style().standardIcon(QStyle.SP_FileDialogDetailedView), GetFromLanguage("MenuReports", 3), self,
                                               triggered=self.generate_clustering_report)

    def create_menu(self):
        """Creates the main menu bar."""
        menubar = self.menuBar()
        file_menu = menubar.addMenu(GetFromLanguage("MenuFile", 0))
        file_menu.addAction(self.import_action)
        file_menu.addAction(self.export_excel_action)
        file_menu.addSeparator()
        file_menu.addAction(self.clear_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        file_menu.addSeparator()
        file_menu.addAction(self.change_language)

        tools_menu = menubar.addMenu(GetFromLanguage("MenuTools", 0))
        tools_menu.addAction(self.calc_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.settings_action)
        
        reports_menu = menubar.addMenu(GetFromLanguage("MenuReports", 0))
        reports_menu.addAction(self.indices_report_action)
        reports_menu.addAction(self.statistics_report_action)
        reports_menu.addAction(self.clustering_report_action)

    def create_toolbar(self):
        """Creates the main toolbar."""
        toolbar = QToolBar(GetFromLanguage("Toolbar", 0))
        toolbar.setIconSize(QSize(32, 32))  # Larger icons
        self.addToolBar(toolbar)
        toolbar.addAction(self.import_action)
        toolbar.addAction(self.clear_action)
        toolbar.addSeparator()
        toolbar.addAction(self.calc_action)
        toolbar.addSeparator()
        toolbar.addAction(self.export_excel_action)
        toolbar.addSeparator()
        toolbar.addAction(self.change_language)

    def get_professional_group_style(self):
        """Retorna um estilo CSS profissional para QGroupBox."""
        return """
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #333;
                border: 1px solid #D0D0D0;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                background-color: #F0F0F0;
                border-radius: 4px;
            }
        """

    def get_professional_button_style(self):
        """Retorna um estilo CSS profissional para botões."""
        return """
            QPushButton {
                background-color: #4CAF50; /* Verde */
                color: white;
                padding: 10px 15px;
                font-weight: bold;
                border-radius: 5px;
                border: 1px solid #388E3C;
            }
            QPushButton:hover {
                background-color: #45a049; /* Verde mais escuro */
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
                border-color: #999999;
            }
        """

    #TODO: Analizar essa função
    def toggle_report_sections(self, state):
        """Marca ou desmarca todas as checkboxes de seção do relatório."""
        check_state = Qt.Checked if state == Qt.Checked else Qt.Unchecked
        
        # Itera sobre todas as checkboxes que são filhas dos group boxes
        for group_box in [self.basic_options_group, self.regression_options_group, self.pca_options_group]:
            if hasattr(self, 'basic_options_group') and self.basic_options_group is not None:
                checkboxes = group_box.findChildren(QCheckBox)
                for checkbox in checkboxes:
                    # Não alterar o estado do "Selecionar Todos"
                    if "Todos" not in checkbox.text():
                        checkbox.setCheckState(check_state)

    def create_reports_tab(self):
        """Cria a aba de relatórios com layout e estilo aprimorados."""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # --- Título e Descrição ---
        title_label = QLabel(GetFromLanguage("ReportsTab", 0))
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #333;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        info_label = QLabel(GetFromLanguage("ReportsTab", 1))
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("font-size: 14px; color: #666; margin-bottom: 20px;")
        main_layout.addWidget(info_label)

        # --- Layout Principal (Duas Colunas) ---
        content_layout = QHBoxLayout()
        left_column = QVBoxLayout()
        right_column = QVBoxLayout()
        
        content_layout.addLayout(left_column, 2)  # Coluna de opções com 2/3 do espaço
        content_layout.addLayout(right_column, 1) # Coluna de geração com 1/3 do espaço

        # --- Coluna Esquerda: Opções de Conteúdo ---
        options_container = QGroupBox(GetFromLanguage("ReportsTab", 2))
        options_container.setStyleSheet(self.get_professional_group_style())
        options_layout = QVBoxLayout(options_container)
        
        # Botões de Seleção Rápida
        selection_buttons_layout = QHBoxLayout()
        select_all_button = QPushButton(GetFromLanguage("ReportsTab", 3))
        deselect_all_button = QPushButton(GetFromLanguage("ReportsTab", 4))
        select_all_button.setStyleSheet("padding: 5px;")
        deselect_all_button.setStyleSheet("padding: 5px;")
        
        selection_buttons_layout.addWidget(select_all_button)
        selection_buttons_layout.addWidget(deselect_all_button)
        selection_buttons_layout.addStretch()
        options_layout.addLayout(selection_buttons_layout)

        # Agrupamento das checkboxes
        self.basic_options_group = QGroupBox(GetFromLanguage("ReportsTab", 5))
        self.regression_options_group = QGroupBox(GetFromLanguage("ReportsTab", 6))
        self.pca_options_group = QGroupBox(GetFromLanguage("ReportsTab", 7))

        # Layouts para os grupos
        basic_layout = QVBoxLayout(self.basic_options_group)
        regression_layout = QVBoxLayout(self.regression_options_group)
        pca_layout = QVBoxLayout(self.pca_options_group)

        # Checkboxes de Análises Básicas
        self.include_descriptive = QCheckBox(GetFromLanguage("ReportsTab", 8))
        self.include_correlation = QCheckBox(GetFromLanguage("ReportsTab", 9))
        self.include_normality = QCheckBox(GetFromLanguage("ReportsTab", 10))
        self.include_outliers = QCheckBox(GetFromLanguage("ReportsTab", 11))
        self.include_clustering = QCheckBox(GetFromLanguage("ReportsTab", 12))
        
        basic_checkboxes = [self.include_descriptive, self.include_correlation, self.include_normality, self.include_outliers, self.include_clustering]
        for cb in basic_checkboxes:
            cb.setChecked(True)
            basic_layout.addWidget(cb)

        # Checkboxes de Regressão
        self.include_regression_summary = QCheckBox(GetFromLanguage("ReportsTab", 13))
        self.include_regression_equation = QCheckBox(GetFromLanguage("ReportsTab", 14))
        self.include_regression_coefficients = QCheckBox(GetFromLanguage("ReportsTab", 15))
        self.include_regression_diagnostics = QCheckBox(GetFromLanguage("ReportsTab", 16))
        self.include_regression_residuals = QCheckBox(GetFromLanguage("ReportsTab", 17))
        self.include_regression_values = QCheckBox(GetFromLanguage("ReportsTab", 18))

        regression_checkboxes = [self.include_regression_summary, self.include_regression_equation, self.include_regression_coefficients, self.include_regression_diagnostics, self.include_regression_residuals, self.include_regression_values]
        for cb in regression_checkboxes:
            cb.setChecked(True)
            regression_layout.addWidget(cb)

        # Checkboxes de PCA
        self.include_pca_eigenvalues = QCheckBox(GetFromLanguage("ReportsTab", 19))
        self.include_pca_eigenvectors = QCheckBox(GetFromLanguage("ReportsTab", 20))
        self.include_pca_loadings = QCheckBox(GetFromLanguage("ReportsTab", 21))
        self.include_pca_coordinates = QCheckBox(GetFromLanguage("ReportsTab", 22))

        pca_checkboxes = [self.include_pca_eigenvalues, self.include_pca_eigenvectors, self.include_pca_loadings, self.include_pca_coordinates]
        for cb in pca_checkboxes:
            cb.setChecked(True)
            pca_layout.addWidget(cb)

        # Adicionar grupos ao layout de opções
        options_layout.addWidget(self.basic_options_group)
        options_layout.addWidget(self.regression_options_group)
        options_layout.addWidget(self.pca_options_group)
        
        left_column.addWidget(options_container)
        left_column.addStretch()

        # --- Coluna Direita: Geração do Relatório ---
        generation_container = QGroupBox(GetFromLanguage("ReportsTab", 23))
        generation_container.setStyleSheet(self.get_professional_group_style())
        generation_layout = QFormLayout(generation_container)
        generation_layout.setRowWrapPolicy(QFormLayout.WrapAllRows)

        # Formato de Saída
        self.report_format = QComboBox()
        self.report_format.addItems([GetFromLanguage("ReportsTab", 27)])
        self.report_format.setStyleSheet("padding: 5px;")
        generation_layout.addRow(QLabel(GetFromLanguage("ReportsTab", 24)), self.report_format)

        # Botão de Geração
        self.generate_report_btn = QPushButton(GetFromLanguage("ReportsTab", 25))
        self.generate_report_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.generate_report_btn.setStyleSheet(self.get_professional_button_style())
        self.generate_report_btn.clicked.connect(self.generate_custom_report)
        
        # Status do Relatório
        self.report_status_label = QLabel(GetFromLanguage("ReportsTab", 26))
        self.report_status_label.setWordWrap(True)
        self.report_status_label.setStyleSheet("font-size: 12px; color: #777; margin-top: 15px;")

        right_column.addWidget(generation_container)
        right_column.addSpacing(20)
        right_column.addWidget(self.generate_report_btn)
        right_column.addWidget(self.report_status_label)
        right_column.addStretch()

        # Conectar botões de seleção
        select_all_button.clicked.connect(lambda: self.toggle_report_sections(Qt.Checked))
        deselect_all_button.clicked.connect(lambda: self.toggle_report_sections(Qt.Unchecked))

        main_layout.addLayout(content_layout)
        return tab

    def generate_custom_report(self):
        """Gera relatório personalizado em Excel."""
        if self.indices.empty:
            QMessageBox.warning(self, GetFromLanguage("Dialogs", 0), GetFromLanguage("Dialogs", 1))
            return
            
        # Escolher arquivo de saída
        file_path, _ = QFileDialog.getSaveFileName(
            self, GetFromLanguage("Dialogs", 2), "relatorio_ipastic.xlsx", GetFromLanguage("Dialogs", 3))
        if file_path:
            self._generate_excel_report(file_path)
    
    def _generate_excel_report(self, file_path):
        """Gera relatório em Excel."""
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Dados de entrada
                self.data.to_excel(writer, sheet_name='1. Dados de Entrada', index=False)
                
                # Índices calculados
                self.indices.to_excel(writer, sheet_name='2. Índices Calculados', index=False)
                
                # Rankings
                if not self.rankings_df.empty:
                    self.rankings_df.to_excel(writer, sheet_name='3. Rankings', index=False)
                
                # Análises estatísticas
                if self.include_descriptive.isChecked():
                    try:
                        data_source = self.indices.select_dtypes(include=np.number)
                        desc_stats = self.analysis.calculate_descriptive_statistics(data_source)
                        if not desc_stats.empty:
                            desc_stats.to_excel(writer, sheet_name='4. Estatísticas Descritivas')
                    except Exception as e:
                        logging.warning(f"Failed to add descriptive stats: {e}")
                
                if self.include_correlation.isChecked():
                    try:
                        data_source = self.indices.select_dtypes(include=np.number)
                        corr_matrix = data_source.corr()
                        if not corr_matrix.empty:
                            corr_matrix.to_excel(writer, sheet_name='5. Matriz de Correlação')
                    except Exception as e:
                        logging.warning(f"Failed to add correlation matrix: {e}")
                
                if self.include_normality.isChecked():
                    try:
                        data_source = self.indices.select_dtypes(include=np.number)
                        norm_results = self.analysis.perform_normality_tests(data_source)
                        if not norm_results.empty:
                            norm_results.to_excel(writer, sheet_name='6. Testes de Normalidade', index=False)
                    except Exception as e:
                        logging.warning(f"Failed to add normality tests: {e}")
                
                if self.include_outliers.isChecked():
                    try:
                        outliers_results = self.analysis.detect_outliers(self.indices)
                        if not outliers_results.empty:
                            outliers_results.to_excel(writer, sheet_name='7. Detecção de Outliers', index=False)
                    except Exception as e:
                        logging.warning(f"Failed to add outliers: {e}")
                
                # Adicionar seções de regressão se disponíveis
                self._add_regression_sections_to_excel(writer)
                
                # Adicionar seções de PCA se disponíveis
                self._add_pca_sections_to_excel(writer)
                
            
            self.report_status_label.setText(f"✅ Relatório Excel gerado com sucesso: {file_path}")
            self.status_bar.showMessage("Relatório Excel gerado!", 5000)
            
        except Exception as e:
            QMessageBox.critical(self, "Erro ao Gerar Relatório", 
                               f"Erro ao gerar relatório Excel:\n{str(e)}")
            self.report_status_label.setText("❌ Erro ao gerar relatório Excel")
    
    def _add_regression_sections_to_excel(self, writer):
        """Adiciona seções de regressão ao relatório Excel."""
        try:
            import pandas as pd
            # Verificar se existem resultados de regressão
            print(f"DEBUG: Verificando resultados de regressão...")
            print(f"DEBUG: hasattr current_regression_results: {hasattr(self, 'current_regression_results')}")
            if hasattr(self, 'current_regression_results'):
                print(f"DEBUG: current_regression_results: {self.current_regression_results is not None}")
            
            if not hasattr(self, 'current_regression_results') or not self.current_regression_results:
                print("DEBUG: Não há resultados de regressão disponíveis")
                return
            
            print("DEBUG: Adicionando seções de regressão ao Excel...")
                
            results = self.current_regression_results
            model = results['model']
            
            # 1. Resumo do Modelo
            if self.include_regression_summary.isChecked():
                print("DEBUG: Criando resumo do modelo...")
                try:
                    summary_data = []
                    r2_pct = model.rsquared * 100
                    r2_adj_pct = model.rsquared_adj * 100
                    
                    summary_data.append(['Métrica', 'Valor'])
                    summary_data.append(['R² (R-squared)', f'{r2_pct:.4f}%'])
                    summary_data.append(['R² Ajustado (Adj R-squared)', f'{r2_adj_pct:.4f}%'])
                    summary_data.append(['Estatística F', f'{model.fvalue:.4f}'])
                    summary_data.append(['Valor-p F', f'{model.f_pvalue:.4f}'])
                    summary_data.append(['AIC', f'{model.aic:.4f}'])
                    summary_data.append(['BIC', f'{model.bic:.4f}'])
                    summary_data.append(['Log-Likelihood', f'{model.llf:.4f}'])
                    summary_data.append(['Número de Observações', f'{int(results["n_obs"])}'])
                    
                    df_summary = pd.DataFrame(summary_data[1:], columns=summary_data[0])
                    df_summary.to_excel(writer, sheet_name='Reg_1_Resumo', index=False)
                    print("DEBUG: Planilha de resumo criada com sucesso!")
                except Exception as e:
                    print(f"DEBUG: Erro ao criar resumo: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 2. Equação do Modelo
            if self.include_regression_equation.isChecked():
                print("DEBUG: Criando equação do modelo...")
                equation = results.get('equation', 'Equação não disponível')
                print(f"DEBUG: Equação obtida: {equation}")
                equation_data = [['Equação do Modelo'], [equation]]
                df_equation = pd.DataFrame(equation_data[1:], columns=equation_data[0])
                df_equation.to_excel(writer, sheet_name='Reg_2_Equacao', index=False)
                print("DEBUG: Planilha de equação criada com sucesso!")
            
            # 3. Coeficientes e Significância
            if self.include_regression_coefficients.isChecked():
                print("DEBUG: Criando coeficientes e significância...")
                try:
                    coef_data = []
                    coef_data.append(['Variável', 'Coeficiente', 'Erro Padrão', 'Estatística t', 'Valor-p', 'IC 95% Inferior', 'IC 95% Superior'])
                    
                    # Obter intervalos de confiança
                    conf_int = model.conf_int()
                    print(f"DEBUG: Intervalos de confiança: {conf_int}")
                    
                    # Obter nomes dos parâmetros
                    if hasattr(model.params, 'index'):
                        param_names = list(model.params.index)
                        params = model.params
                        bse = model.bse
                        tvalues = model.tvalues
                        pvalues = model.pvalues
                    else:
                        # Se params é numpy array, usar var_names dos resultados
                        param_names = results.get('var_names', [f'Var_{i}' for i in range(len(model.params))])
                        params = model.params
                        bse = model.bse if hasattr(model.bse, '__len__') else [model.bse] * len(params)
                        tvalues = model.tvalues if hasattr(model.tvalues, '__len__') else [model.tvalues] * len(params)
                        pvalues = model.pvalues if hasattr(model.pvalues, '__len__') else [model.pvalues] * len(params)
                    
                    print(f"DEBUG: param_names: {param_names}")
                    print(f"DEBUG: params type: {type(params)}")
                    
                    for i, param_name in enumerate(param_names):
                        if hasattr(params, 'iloc'):
                            coef = params.iloc[i] if i < len(params) else 0
                            std_err = bse.iloc[i] if i < len(bse) else 0
                            t_stat = tvalues.iloc[i] if i < len(tvalues) else 0
                            p_val = pvalues.iloc[i] if i < len(pvalues) else 0
                        else:
                            coef = params[i] if i < len(params) else 0
                            std_err = bse[i] if i < len(bse) else 0
                            t_stat = tvalues[i] if i < len(tvalues) else 0
                            p_val = pvalues[i] if i < len(pvalues) else 0
                        
                        # Intervalos de confiança
                        if hasattr(conf_int, 'iloc'):
                            ci_lower = conf_int.iloc[i, 0] if i < len(conf_int) else 0
                            ci_upper = conf_int.iloc[i, 1] if i < len(conf_int) else 0
                        else:
                            ci_lower = conf_int[i, 0] if i < len(conf_int) else 0
                            ci_upper = conf_int[i, 1] if i < len(conf_int) else 0
                        
                        coef_data.append([
                            param_name, 
                            f'{coef:.6f}',
                            f'{std_err:.6f}',
                            f'{t_stat:.6f}',
                            f'{p_val:.6f}',
                            f'{ci_lower:.6f}',
                            f'{ci_upper:.6f}'
                        ])
                    
                    df_coef = pd.DataFrame(coef_data[1:], columns=coef_data[0])
                    df_coef.to_excel(writer, sheet_name='Reg_3_Coeficientes', index=False)
                    print("DEBUG: Planilha de coeficientes criada com sucesso!")
                except Exception as e:
                    print(f"DEBUG: Erro ao criar coeficientes: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 4. Diagnósticos do Modelo
            if self.include_regression_diagnostics.isChecked():
                print("DEBUG: Criando diagnósticos do modelo...")
                try:
                    diag_data = []
                    diag_data.append(['Estatística/Teste', 'Valor', 'Interpretação'])
                    
                    # Estatísticas básicas do modelo
                    diag_data.append(['Número de observações', f'{int(results["n_obs"])}', 'Tamanho da amostra'])
                    diag_data.append(['Graus de liberdade (modelo)', f'{model.df_model:.1f}', 'Parâmetros estimados'])
                    diag_data.append(['Graus de liberdade (resíduos)', f'{model.df_resid:.1f}', 'Obs - parâmetros'])
                    diag_data.append(['R-quadrado', f'{model.rsquared:.6f}', 'Variação explicada'])
                    diag_data.append(['R-quadrado ajustado', f'{model.rsquared_adj:.6f}', 'R² ajustado pelo n° de variáveis'])
                    
                    # Calcular raiz do MSE
                    import numpy as np
                    mse_resid = model.mse_resid
                    rmse = np.sqrt(mse_resid)
                    diag_data.append(['Raiz do MSE', f'{rmse:.6f}', 'Erro padrão dos resíduos'])
                    
                    diag_data.append(['AIC', f'{model.aic:.4f}', 'Critério de informação de Akaike'])
                    diag_data.append(['BIC', f'{model.bic:.4f}', 'Critério de informação bayesiano'])
                    diag_data.append(['Log-Likelihood', f'{model.llf:.4f}', 'Log-verossimilhança'])
                    diag_data.append(['F-estatística', f'{model.fvalue:.4f}', 'Teste F do modelo'])
                    diag_data.append(['P-valor (F)', f'{model.f_pvalue:.4e}', 'Significância do modelo'])
                    diag_data.append(['Condição do modelo', f'{model.condition_number:.2f}', 'Multicolinearidade'])
                    
                    # Teste de homocedasticidade (Breusch-Pagan)
                    try:
                        from statsmodels.stats.diagnostic import het_breuschpagan
                        bp_lm, bp_lm_pval, bp_f, bp_f_pval = het_breuschpagan(model.resid, model.model.exog)
                        diag_data.append(['Breusch-Pagan (estat.)', f'{bp_lm:.4f}', 'Teste de homocedasticidade'])
                        diag_data.append(['Breusch-Pagan (p-valor)', f'{bp_lm_pval:.4e}', 'Homocedástico se > 0.05'])
                    except Exception as bp_e:
                        print(f"DEBUG: Erro no teste Breusch-Pagan: {bp_e}")
                        diag_data.append(['Breusch-Pagan (estat.)', 'Erro no cálculo', 'N/A'])
                        diag_data.append(['Breusch-Pagan (p-valor)', 'Erro no cálculo', 'N/A'])
                    
                    # Teste de normalidade (Jarque-Bera)
                    try:
                        from scipy import stats
                        jb_stat, jb_pval = stats.jarque_bera(model.resid)
                        diag_data.append(['Jarque-Bera (estat.)', f'{jb_stat:.4f}', 'Teste de normalidade'])
                        diag_data.append(['Jarque-Bera (p-valor)', f'{jb_pval:.4e}', 'Normal se > 0.05'])
                    except Exception as jb_e:
                        print(f"DEBUG: Erro no teste Jarque-Bera: {jb_e}")
                        # Fallback - calculate manually using scipy.stats
                        try:
                            import numpy as np
                            from scipy import stats
                            # Manual Jarque-Bera calculation
                            residuals = model.resid
                            n = len(residuals)
                            skewness = stats.skew(residuals)
                            kurtosis = stats.kurtosis(residuals)
                            jb_stat = (n/6) * (skewness**2 + (kurtosis**2)/4)
                            jb_pval = 1 - stats.chi2.cdf(jb_stat, 2)
                            diag_data.append(['Jarque-Bera (estat.)', f'{jb_stat:.4f}', 'Teste de normalidade'])
                            diag_data.append(['Jarque-Bera (p-valor)', f'{jb_pval:.4e}', 'Normal se > 0.05'])
                        except:
                            diag_data.append(['Jarque-Bera (estat.)', 'Erro no cálculo', 'N/A'])
                            diag_data.append(['Jarque-Bera (p-valor)', 'Erro no cálculo', 'N/A'])
                    
                    # Teste de homocedasticidade (Breusch-Pagan)
                    try:
                        from statsmodels.stats.diagnostic import het_breuschpagan
                        bp_lm, bp_lm_pval, bp_f, bp_f_pval = het_breuschpagan(model.resid, model.model.exog)
                        bp_interp = 'Homocedástico' if bp_lm_pval > 0.05 else 'Heterocedástico'
                        diag_data.append(['Breusch-Pagan (Homocedasticidade)', f'LM={bp_lm:.4f}, p={bp_lm_pval:.4f}', bp_interp])
                    except Exception as bp_e:
                        print(f"DEBUG: Erro no teste Breusch-Pagan: {bp_e}")
                        diag_data.append(['Breusch-Pagan (estat.)', 'Erro no cálculo', 'N/A'])
                        diag_data.append(['Breusch-Pagan (p-valor)', 'Erro no cálculo', 'N/A'])
                    
                    # Durbin-Watson (Autocorrelação)
                    try:
                        from statsmodels.stats.stattools import durbin_watson
                        dw_stat = durbin_watson(model.resid)
                        diag_data.append(['Durbin-Watson', f'{dw_stat:.4f}', 'Autocorrelação: normal se 1.5-2.5'])
                    except Exception as dw_e:
                        print(f"DEBUG: Erro no teste Durbin-Watson: {dw_e}")
                        diag_data.append(['Durbin-Watson', 'Erro no cálculo', 'N/A'])
                    
                    df_diag = pd.DataFrame(diag_data[1:], columns=diag_data[0])
                    df_diag.to_excel(writer, sheet_name='Reg_4_Diagnosticos', index=False)
                    print("DEBUG: Planilha de diagnósticos criada com sucesso!")
                except Exception as e:
                    print(f"DEBUG: Erro ao criar diagnósticos: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 5. Análise de Resíduos
            if self.include_regression_residuals.isChecked():
                print("DEBUG: Criando análise de resíduos...")
                try:
                    resid_data = []
                    residuals = model.resid
                    
                    resid_data.append(['Estatística', 'Valor'])
                    resid_data.append(['Média dos Resíduos', f'{residuals.mean():.6f}'])
                    resid_data.append(['Desvio Padrão dos Resíduos', f'{residuals.std():.6f}'])
                    resid_data.append(['Resíduo Mínimo', f'{residuals.min():.6f}'])
                    resid_data.append(['Resíduo Máximo', f'{residuals.max():.6f}'])
                    # Usar numpy para quantis se residuals for numpy array
                    import numpy as np
                    if hasattr(residuals, 'quantile'):
                        q25 = residuals.quantile(0.25)
                        median = residuals.median()
                        q75 = residuals.quantile(0.75)
                    else:
                        q25 = np.percentile(residuals, 25)
                        median = np.median(residuals)
                        q75 = np.percentile(residuals, 75)
                    
                    resid_data.append(['Quartil 25%', f'{q25:.6f}'])
                    resid_data.append(['Mediana', f'{median:.6f}'])
                    resid_data.append(['Quartil 75%', f'{q75:.6f}'])
                    
                    # Criar planilha com estatísticas descritivas
                    df_resid = pd.DataFrame(resid_data[1:], columns=resid_data[0])
                    df_resid.to_excel(writer, sheet_name='Reg_5_Residuos', index=False)
                    
                    # Adicionar também uma tabela detalhada de resíduos por genótipo
                    residuals_detail_data = []
                    residuals_detail_data.append(['Genótipo', 'Resíduo', 'Resíduo Padronizado', 'Resíduo Absoluto'])
                    
                    # Obter nomes dos genótipos usando a mesma lógica
                    genotype_col = results.get('genotype_col', 'Genótipo')
                    data_copy = results['data_copy']
                    
                    residuals_array = residuals.values if hasattr(residuals, 'values') else residuals
                    std_residuals = np.std(residuals_array)
                    
                    for i in range(len(residuals_array)):
                        # Usar nomes reais dos genótipos se disponível
                        if genotype_col and genotype_col in data_copy.columns and i < len(data_copy):
                            genotype = str(data_copy.iloc[i][genotype_col])
                        else:
                            # Fallback para numeração sequencial
                            genotype = f'{i+1}'
                        
                        resid_val = residuals_array[i]
                        resid_std = resid_val / std_residuals if std_residuals != 0 else 0
                        resid_abs = abs(resid_val)
                        
                        residuals_detail_data.append([
                            genotype,
                            f'{resid_val:.6f}',
                            f'{resid_std:.6f}',
                            f'{resid_abs:.6f}'
                        ])
                    
                    # Criar DataFrame e adicionar à planilha existente (na mesma aba, após as estatísticas)
                    worksheet = writer.sheets['Reg_5_Residuos']
                    
                    # Adicionar uma linha em branco
                    start_row = len(df_resid) + 3
                    worksheet.write(start_row, 0, "Resíduos por Genótipo")
                    
                    # Adicionar cabeçalhos da tabela detalhada
                    for col, header in enumerate(residuals_detail_data[0]):
                        worksheet.write(start_row + 2, col, header)
                    
                    # Adicionar dados da tabela detalhada
                    for row_idx, row_data in enumerate(residuals_detail_data[1:], start=start_row + 3):
                        for col_idx, value in enumerate(row_data):
                            worksheet.write(row_idx, col_idx, value)
                    
                    print("DEBUG: Planilha de resíduos com estatísticas e valores individuais criada com sucesso!")
                except Exception as e:
                    print(f"DEBUG: Erro ao criar análise de resíduos: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 6. Valores Observados vs Estimados
            if self.include_regression_values.isChecked():
                print("DEBUG: Criando valores observados vs estimados...")
                try:
                    y_var = results['y_var']
                    genotype_col = results.get('genotype_col', 'Genótipo')
                    print(f"DEBUG: y_var={y_var}, genotype_col={genotype_col}")
                    
                    # Obter dados originais
                    data_copy = results['data_copy']
                    y_observed = data_copy[y_var]
                    y_predicted = model.fittedvalues
                    residuals = model.resid
                    
                    print(f"DEBUG: Tamanhos - observed: {len(y_observed)}, predicted: {len(y_predicted)}, residuals: {len(residuals)}")
                    
                    values_data = []
                    values_data.append(['Genótipo', 'Valor Observado', 'Valor Estimado', 'Resíduo', 'Resíduo %', 'Resíduo Absoluto'])
                    
                    # Criar lista única de dados para evitar duplicação
                    data_rows = []
                    n_obs = len(y_observed)
                    
                    for i in range(n_obs):
                        # Usar nomes reais dos genótipos se disponível
                        if genotype_col and genotype_col in data_copy.columns and i < len(data_copy):
                            genotype = str(data_copy.iloc[i][genotype_col])
                        else:
                            # Fallback para numeração sequencial simples: 1, 2, 3, ..., 90
                            genotype = f'{i+1}'
                        
                        # Lidar com numpy arrays e pandas Series
                        if hasattr(y_observed, 'iloc'):
                            obs = y_observed.iloc[i]
                        else:
                            obs = y_observed[i]
                        
                        if hasattr(y_predicted, 'iloc'):
                            pred = y_predicted.iloc[i]
                        else:
                            pred = y_predicted[i]
                        
                        if hasattr(residuals, 'iloc'):
                            resid = residuals.iloc[i]
                        else:
                            resid = residuals[i]
                        
                        resid_pct = (resid / obs * 100) if obs != 0 else 0
                        resid_abs = abs(resid)
                        
                        data_rows.append({
                            'genotype': genotype,
                            'obs': obs,
                            'pred': pred,
                            'resid': resid,
                            'resid_pct': resid_pct,
                            'resid_abs': resid_abs,
                            'index': i
                        })
                    
                    # Ordenar por ordem crescente (sequencial 1, 2, 3, ...)
                    data_rows.sort(key=lambda x: int(x['genotype']))
                    
                    # Adicionar linhas ordenadas
                    for row in data_rows:
                        values_data.append([
                            row['genotype'],
                            f'{row["obs"]:.4f}',
                            f'{row["pred"]:.4f}',
                            f'{row["resid"]:.4f}',
                            f'{row["resid_pct"]:.2f}%',
                            f'{row["resid_abs"]:.4f}'
                        ])
                    
                    df_values = pd.DataFrame(values_data[1:], columns=values_data[0])
                    df_values.to_excel(writer, sheet_name='Reg_6_Valores', index=False)
                    print("DEBUG: Planilha de valores observados vs estimados criada com sucesso!")
                except Exception as e:
                    print(f"DEBUG: Erro ao criar valores observados vs estimados: {e}")
                    import traceback
                    traceback.print_exc()
                
            print("DEBUG: Todas as seções de regressão processadas com sucesso!")
                
        except Exception as e:
            print(f"DEBUG: Erro geral na função de regressão: {e}")
            import traceback
            traceback.print_exc()
            logging.warning(f"Erro ao adicionar seções de regressão ao Excel: {e}")
    
    def _add_pca_sections_to_excel(self, writer):
        """Adiciona seções de PCA ao relatório Excel."""
        try:
            # Verificar se existem resultados de PCA
            print(f"DEBUG PCA: Verificando resultados de PCA...")
            if not hasattr(self, 'current_pca_results') or not self.current_pca_results:
                print("DEBUG PCA: Não há resultados de PCA disponíveis")
                return
                
            results = self.current_pca_results
            print(f"DEBUG PCA: Adicionando seções de PCA ao Excel com {results['n_components']} componentes...")
            
            # Limitar a 10 componentes conforme solicitado
            max_components = min(results['n_components'], 10)
            
            # 1. Autovalores e Variância Explicada
            if self.include_pca_eigenvalues.isChecked():
                eigenvalues_data = []
                eigenvalues_data.append(['Componente', 'Autovalor', 'Variância Explicada (%)', 'Variância Acumulada (%)'])
                
                cumulative_variance = 0
                for i in range(max_components):
                    eigenvalue = results['eigenvalues'][i]
                    variance_pct = results['explained_variance_ratio'][i] * 100
                    cumulative_variance += variance_pct
                    
                    eigenvalues_data.append([
                        f'PC{i+1}',
                        f'{eigenvalue:.6f}',
                        f'{variance_pct:.4f}%',
                        f'{cumulative_variance:.4f}%'
                    ])
                
                df_eigenvalues = pd.DataFrame(eigenvalues_data[1:], columns=eigenvalues_data[0])
                df_eigenvalues.to_excel(writer, sheet_name='PCA_1_Autovalores', index=False)
            
            # 2. Autovetores (Componentes Principais)
            if self.include_pca_eigenvectors.isChecked():
                eigenvectors_data = []
                header = ['Variável'] + [f'PC{i+1}' for i in range(max_components)]
                eigenvectors_data.append(header)
                
                for j, variable in enumerate(results['feature_names']):
                    row = [variable]
                    for i in range(max_components):
                        row.append(f'{results["eigenvectors"][i][j]:.6f}')
                    eigenvectors_data.append(row)
                
                df_eigenvectors = pd.DataFrame(eigenvectors_data[1:], columns=eigenvectors_data[0])
                df_eigenvectors.to_excel(writer, sheet_name='PCA_2_Autovetores', index=False)
            
            # 3. Cargas Fatoriais (Loadings)
            if self.include_pca_loadings.isChecked():
                loadings_data = []
                header = ['Variável'] + [f'PC{i+1}' for i in range(max_components)]
                loadings_data.append(header)
                
                for j, variable in enumerate(results['feature_names']):
                    row = [variable]
                    for i in range(max_components):
                        row.append(f'{results["loadings"][j][i]:.6f}')
                    loadings_data.append(row)
                
                df_loadings = pd.DataFrame(loadings_data[1:], columns=loadings_data[0])
                df_loadings.to_excel(writer, sheet_name='PCA_3_Loadings', index=False)
            
            # 4. Coordenadas dos Indivíduos
            if self.include_pca_coordinates.isChecked():
                coords_data = []
                header = ['Indivíduo'] + [f'PC{i+1}' for i in range(max_components)]
                coords_data.append(header)
                
                # Usar nomes dos genótipos diretamente do índice das coordenadas
                coordinates = results['coordinates']
                
                for i in range(len(coordinates)):
                    # Usar o índice das coordenadas que já contém os nomes reais dos genótipos
                    individual_name = str(coordinates.index[i])
                    
                    row = [individual_name]
                    for j in range(max_components):
                        row.append(f'{coordinates.iloc[i, j]:.6f}')
                    coords_data.append(row)
                
                df_coords = pd.DataFrame(coords_data[1:], columns=coords_data[0])
                df_coords.to_excel(writer, sheet_name='PCA_4_Coordenadas', index=False)
                
            print(f"DEBUG PCA: Seções de PCA adicionadas com sucesso!")
                
        except Exception as e:
            logging.warning(f"Erro ao adicionar seções de PCA ao Excel: {e}")
            print(f"DEBUG PCA: Erro - {e}")
    







    def create_input_tab(self):
        """Creates the data input tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Table for data input
        self.table_input = QTableWidget(15, 3) # Start with 15 rows, 3 columns
        self.table_input.setHorizontalHeaderLabels(self.original_column_names)
        self.table_input.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_input.itemChanged.connect(self.on_table_item_changed)
        
        # Enable context menu for copy/paste
        self.table_input.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_input.customContextMenuRequested.connect(lambda pos: self.show_table_context_menu(pos, self.table_input))

        # Add row/column buttons
        btn_layout = QHBoxLayout()
        add_row_btn = QPushButton(GetFromLanguage("InputTab", 0))
        add_row_btn.clicked.connect(lambda: self.table_input.insertRow(self.table_input.rowCount()))
        remove_row_btn = QPushButton(GetFromLanguage("InputTab", 1))
        remove_row_btn.clicked.connect(self.remove_selected_row)
        btn_layout.addWidget(add_row_btn)
        btn_layout.addWidget(remove_row_btn)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)
        layout.addWidget(self.table_input)
        return tab

    def show_table_context_menu(self, pos, table_widget):
        """Shows a context menu for the given table widget."""
        context_menu = QMenu(self)
        copy_action = context_menu.addAction(GetFromLanguage("InputTab", 2))
        
        paste_action = None
        if table_widget == self.table_input:
            paste_action = context_menu.addAction(GetFromLanguage("InputTab", 3))

        action = context_menu.exec_(table_widget.mapToGlobal(pos))
        
        if action == copy_action:
            self.copy_table_selection(table_widget)
        elif action == paste_action:
            self.paste_into_input_table()

    def copy_table_selection(self, table_widget):
        """Copies the selected cells from a generic table widget to the clipboard."""
        selection = table_widget.selectedRanges()
        if not selection:
            return

        rows = []
        # Get column headers if they exist
        if table_widget.horizontalHeader().count() > 0:
            header_items = []
            for c in range(selection[0].leftColumn(), selection[0].rightColumn() + 1):
                item = table_widget.horizontalHeaderItem(c)
                header_items.append(item.text() if item else "")
            rows.append("\t".join(header_items))

        for r in range(selection[0].topRow(), selection[0].bottomRow() + 1):
            row_data = []
            for c in range(selection[0].leftColumn(), selection[0].rightColumn() + 1):
                item = table_widget.item(r, c)
                row_data.append(item.text() if item else "")
            rows.append("\t".join(row_data))
        
        clipboard = QApplication.clipboard()
        clipboard.setText("\n".join(rows))

    def paste_into_input_table(self):
        """Pastes data from the clipboard into the input table."""
        # This function remains specific to the input table
        selection = self.table_input.selectedRanges()
        if not selection:
            # If no selection, paste from the top-left cell
            start_row = 0
            start_col = 0
        else:
            start_row = selection[0].topRow()
            start_col = selection[0].leftColumn()

        clipboard = QApplication.clipboard()
        text = clipboard.text()
        rows = text.split('\n')

        for i, row_text in enumerate(rows):
            if not row_text:
                continue
            
            values = row_text.split('\t')
            for j, value in enumerate(values):
                row = start_row + i
                col = start_col + j
                
                # Add new rows if paste exceeds current row count
                if row >= self.table_input.rowCount():
                    self.table_input.insertRow(row)

                self.table_input.setItem(row, col, QTableWidgetItem(value))

    def create_indices_tab(self):
        """Creates the calculated indices tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.table_indices = QTableWidget()
        self.table_indices.setSortingEnabled(True)
        self.table_indices.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Add context menu for copying
        self.table_indices.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_indices.customContextMenuRequested.connect(
            lambda pos: self.show_table_context_menu(pos, self.table_indices)
        )
        
        layout.addWidget(self.table_indices)
        return tab

    def create_rankings_tab(self):
        """Creates the rankings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.table_rankings = QTableWidget()
        self.table_rankings.setSortingEnabled(True)
        self.table_rankings.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Add context menu for copying
        self.table_rankings.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_rankings.customContextMenuRequested.connect(
            lambda pos: self.show_table_context_menu(pos, self.table_rankings)
        )

        layout.addWidget(self.table_rankings)
        return tab

"""
iPASTIC Desktop v2.5
Single-file application for calculating plant stress tolerance indices,
performing cluster analysis, and visualizing results.
"""

import sys
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Import PyQt5 components
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
                             QMenuBar, QToolBar, QAction, QMessageBox, QFileDialog, QInputDialog, QDockWidget,
                             QStatusBar, QComboBox, QPushButton, QLabel, QLineEdit, QFormLayout, QHeaderView,
                             QStyleFactory, QStyle, QColorDialog, QMenu, QShortcut, QKeySequenceEdit, QHBoxLayout, QDialog,
                             QProgressBar, QSpinBox, QSplitter, QGroupBox, QCheckBox, QTextEdit, QListWidget, QScrollArea)
from PyQt5.QtGui import QIcon, QPixmap, QFont, QColor, QKeySequence, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, pyqtSlot, QSize, QTimer

# Core scientific computing
import warnings
warnings.filterwarnings('ignore')

# Import enhanced analyses
try:
    from enhanced_analyses import EnhancedAnalyses
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    logging.warning("Enhanced analyses not available. Some advanced features will be disabled.")

# Plotting - defer backend selection to avoid Qt widget creation before QApplication
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Statistical analysis
import scipy.stats as stats
from scipy.cluster.hierarchy import dendrogram, linkage
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera

# Statistical libraries for advanced analysis (optional imports, handled in methods)
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Some advanced features will be disabled.")

# Report generation (optional imports, handled in methods)
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    logging.warning("openpyxl not available. Excel export will be disabled.")

# Interactive plotting (optional imports)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Interactive plots will be disabled.")

# Navigation toolbar for matplotlib
try:
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    NAVIGATION_AVAILABLE = True
except ImportError:
    NAVIGATION_AVAILABLE = False
    logging.warning("Navigation toolbar not available.")

# Check for required dependencies and provide helpful error messages
MISSING_DEPENDENCIES = []

if not SKLEARN_AVAILABLE:
    MISSING_DEPENDENCIES.append("scikit-learn (para PCA e clustering)")
if not PLOTLY_AVAILABLE:
    MISSING_DEPENDENCIES.append("plotly (para gráficos interativos)")
if not OPENPYXL_AVAILABLE:
    MISSING_DEPENDENCIES.append("openpyxl (para exportação Excel)")

#TODO: FIX RETURN
def check_dependency_for_feature(feature_name, dependency_flag, dependency_name):
    """Check if a dependency is available for a specific feature."""
    if not dependency_flag:
        raise ImportError(f"{feature_name} requer {dependency_name}. "
                         f"Execute: pip install {dependency_name.split()[0]}")
    return True

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Configuration ---

# Configure logging to file
logging.basicConfig(filename='app.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Business Logic and Data Processing Classes ---

class IpasticAnalysis:

    def detect_outliers(self, indices):
        """Detecta outliers usando Z-score e IQR."""
        results = []
        if indices.empty:
            return pd.DataFrame()
        numeric_cols = indices.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            data = indices[col].dropna()
            # Z-score
            z_scores = np.abs(stats.zscore(data))
            outliers_z = data[z_scores > 3].index.tolist()
            # IQR
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            outliers_iqr = data[(data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)].index.tolist()
            results.append({'Variável': col, 'Outliers Z-score': outliers_z, 'Outliers IQR': outliers_iqr})
        return pd.DataFrame(results)


    #TODO:FIX LIMIT=NONE
    def classificar_tolerancia(self, rp_value, limits=None):
        """
        Classifica o genótipo baseado no valor de RP (Relative Performance).
        Args:
            rp_value (float): Valor do RP (Redução Relativa na Produtividade)
            limits (dict): Limites para classificação (opcional)
        Returns:
            tuple: (classificacao, cor_hex)
        """
        # Classificação conforme Tabela 6 (Fageria, 1985a)
        # 0-20: Tolerante
        # 20,1-40: Moderadamente tolerante
        # 40,1-60: Moderadamente suscetível
        # >60: Suscetível
        if pd.isna(rp_value):
            return (GetFromLanguage("Tolerance",0), "#CCCCCC")
        if rp_value <= 20:
            return (GetFromLanguage("Tolerance",1), "#4CAF50")  # Verde
        elif rp_value <= 40:
            return (GetFromLanguage("Tolerance", 2), "#8BC34A")  # Verde claro
        elif rp_value <= 60:
            return (GetFromLanguage("Tolerance", 3), "#FF9800")  # Laranja
        else:
            return (GetFromLanguage("Tolerance", 4), "#F44336")  # Vermelho


    #TODO: FIX language
    def calculate_indices(self, data, classification_limits):
        """
        Calculates all stress tolerance indices from the input data.
        Args:
            data (pd.DataFrame): DataFrame with 'Genotipo', 'Yp', and 'Ys' columns.
            classification_limits (dict): Dictionary with tolerance classification thresholds.
        Returns:
            pd.DataFrame: A DataFrame containing the calculated indices for each genotype.
        Raises:
            ValueError: If input data is empty or contains missing values.
        """
        if data.empty or data['Yp'].isnull().any() or data['Ys'].isnull().any():
            raise ValueError("Os dados de entrada para cálculo estão vazios ou contêm valores ausentes em 'Yp' ou 'Ys'.")

        yp_mean = data['Yp'].mean()
        ys_mean = data['Ys'].mean()

        if yp_mean == 0 or ys_mean == 0:
            logging.warning("A média de 'Yp' ou 'Ys' é zero, o que pode levar a divisões por zero.")

        si = 1 - (ys_mean / yp_mean) if yp_mean != 0 else 0
        
        # Preservar ordem original dos genótipos
        # Primeiro criar um mapeamento da ordem original
        genotipo_order = data["Genotipo"].drop_duplicates().tolist()
        
        # Fazer groupby com sort=False para manter ordem original
        indices = data.groupby("Genotipo", sort=False).agg({'Yp': 'mean', 'Ys': 'mean'}).reset_index()
        
        # Se ainda assim a ordem foi alterada, reordenar explicitamente
        if indices["Genotipo"].tolist() != genotipo_order:
            indices['_order'] = indices["Genotipo"].map({g: i for i, g in enumerate(genotipo_order)})
            indices = indices.sort_values('_order').drop('_order', axis=1).reset_index(drop=True)

        # Calculate indices with safeguards against division by zero
        indices['RP'] = ((indices['Yp'] - indices['Ys']) / indices['Yp'].replace(0, np.nan)) * 100
        indices['TOL'] = indices['Yp'] - indices['Ys']
        indices['MP'] = (indices['Yp'] + indices['Ys']) / 2
        indices['GMP'] = np.sqrt(indices['Yp'] * indices['Ys'])
        indices['HMS'] = (2 * indices['Yp'] * indices['Ys']) / (indices['Yp'] + indices['Ys']).replace(0, np.nan)
        indices['SSI'] = (1 - (indices['Ys'] / indices['Yp'].replace(0, np.nan))) / si if si != 0 else 0
        indices['STI'] = (indices['Yp'] * indices['Ys']) / (yp_mean ** 2) if yp_mean != 0 else 0
        indices['YI'] = indices['Ys'] / ys_mean if ys_mean != 0 else 0
        indices['YSI'] = indices['Ys'] / indices['Yp'].replace(0, np.nan)
        indices['RSI'] = (indices['YSI']) * (ys_mean / yp_mean) if yp_mean != 0 else 0

        # Classify and fill NaN values that might have resulted from division by zero
        indices.fillna(0, inplace=True)
        indices['Classificação por RP'], indices['Cor'] = zip(*indices['RP'].apply(self.classificar_tolerancia, limits=classification_limits))
        return indices

    def generate_rankings_df(self, indices):
        """
        Generates rankings for each genotype based on the calculated indices.
        Args:
            indices (pd.DataFrame): DataFrame with calculated indices.
        Returns:
            pd.DataFrame: A DataFrame with a "Rank" column for each index.
        """
        if indices.empty:
            return pd.DataFrame()

        ranking_vars = ['Yp', 'Ys', 'RP', 'TOL', 'MP', 'GMP', 'HMS', 'SSI', 'STI', 'YI', 'YSI', 'RSI']
        # Lower is better for RP, TOL, SSI. Higher is better for the rest.
        ascending_order = {
            'Yp': False, 'Ys': False, 'RP': True, 'TOL': True, 'MP': False,
            'GMP': False, 'HMS': False, 'SSI': True, 'STI': False, 'YI': False,
            'YSI': False, 'RSI': False
        }
        rankings_data = []

        for _, row in indices.iterrows():
            genotype_rankings = {'Genotipo': row['Genotipo'], 'Cor': row['Cor']}
            for var in ranking_vars:
                if var in indices.columns:
                    # Create a rank; 'method' handles ties by assigning the average rank
                    ranks = indices[var].rank(method='min', ascending=ascending_order[var])
                    genotype_rankings[f'Rank_{var}'] = int(ranks[indices['Genotipo'] == row['Genotipo']].iloc[0])
            rankings_data.append(genotype_rankings)

        return pd.DataFrame(rankings_data)

    def perform_clustering(self, indices, method='kmeans', n_clusters='auto', max_clusters=10):
        """
        Performs clustering on the indices with configurable options.
        Args:
            indices (pd.DataFrame): DataFrame with calculated indices.
            method (str): Clustering method ('kmeans' or 'agglomerative').
            n_clusters (int or 'auto'): Number of clusters or 'auto' for optimal selection.
            max_clusters (int): Maximum number of clusters to test when n_clusters='auto'.
        Returns:
            tuple: Contains (cluster_labels, number_of_clusters, silhouette_score).
        """
        if indices.empty or len(indices) < 2:
            raise ValueError("São necessários pelo menos 2 genótipos para realizar a análise de agrupamento.")

        features = indices.select_dtypes(include=np.number).drop(['Cluster'], axis=1, errors='ignore')
        if features.empty:
            raise ValueError("Não há variáveis numéricas para o clustering.")

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        if n_clusters == 'auto':
            # Determine the optimal number of clusters
            max_clusters = min(max_clusters, scaled_features.shape[0] - 1)
            if max_clusters < 2:
                raise ValueError("Não há amostras suficientes para formar mais de um cluster.")

            best_score = -1
            best_n_clusters = 2
            for n_clust in range(2, max_clusters + 1):
                if method == 'kmeans':
                    from sklearn.cluster import KMeans
                    clusterer = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
                elif method == 'agglomerative':
                    from sklearn.cluster import AgglomerativeClustering
                    clusterer = AgglomerativeClustering(n_clusters=n_clust)
                
                clusters = clusterer.fit_predict(scaled_features)
                sil_score = silhouette_score(scaled_features, clusters)
                if sil_score > best_score:
                    best_score = sil_score
                    best_n_clusters = n_clust
            n_clusters = best_n_clusters

        # Final clustering with chosen parameters
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'agglomerative':
            from sklearn.cluster import AgglomerativeClustering
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)

        final_clusters = clusterer.fit_predict(scaled_features)
        final_sil_score = silhouette_score(scaled_features, final_clusters)

        return final_clusters, n_clusters, final_sil_score

    def perform_pca(self, indices, n_components=2, scaling=True):
        """Performs PCA on the indices with configurable options."""
        if indices.empty or len(indices) < 2:
            raise ValueError("PCA requer pelo menos 2 genótipos.")
        features = indices.select_dtypes(include=np.number).drop(['Cluster'], axis=1, errors='ignore')
        
        if scaling:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
        else:
            scaled_features = features.values
            
        # Adjust n_components to available features
        max_components = min(n_components, features.shape[1], features.shape[0])
        pca = PCA(n_components=max_components)
        principal_components = pca.fit_transform(scaled_features)
        
        # Create column names dynamically
        pc_columns = [f'PC{i+1}' for i in range(max_components)]
        pca_df = pd.DataFrame(data=principal_components, columns=pc_columns)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        return pca_df, pca.explained_variance_ratio_, loadings, features.columns, max_components

    def generate_dendrogram_linkage(self, indices, method='ward', metric='euclidean'):
        """Generates the linkage matrix for a dendrogram with configurable method and metric."""
        if indices.empty or len(indices) < 2:
            raise ValueError("Dendrograma requer pelo menos 2 genótipos.")
        features = indices.select_dtypes(include=np.number).drop(['Cluster'], axis=1, errors='ignore')
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        return linkage(scaled_features, method=method, metric=metric)

    def perform_clustering_rankings(self, rankings_df, method='kmeans', n_clusters='auto', max_clusters=10):
        """
        Performs clustering on the ranking values with configurable options.
        Args:
            rankings_df (pd.DataFrame): DataFrame with ranking values.
            method (str): Clustering method ('kmeans' or 'agglomerative').
            n_clusters (int or 'auto'): Number of clusters or 'auto' for optimal selection.
            max_clusters (int): Maximum number of clusters to test when n_clusters='auto'.
        Returns:
            tuple: Contains (cluster_labels, number_of_clusters, silhouette_score).
        """
        if rankings_df.empty or len(rankings_df) < 2:
            raise ValueError("São necessários pelo menos 2 genótipos para realizar a análise de agrupamento.")

        # Select only ranking columns
        rank_features = rankings_df.filter(regex='^Rank_')
        if rank_features.empty:
            raise ValueError("Não há colunas de ranking para o clustering.")

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(rank_features)

        if n_clusters == 'auto':
            # Determine the optimal number of clusters
            max_clusters = min(max_clusters, scaled_features.shape[0] - 1)
            if max_clusters < 2:
                raise ValueError("Não há amostras suficientes para formar mais de um cluster.")

            best_score = -1
            best_n_clusters = 2
            for n_clust in range(2, max_clusters + 1):
                if method == 'kmeans':
                    from sklearn.cluster import KMeans
                    clusterer = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
                elif method == 'agglomerative':
                    from sklearn.cluster import AgglomerativeClustering
                    clusterer = AgglomerativeClustering(n_clusters=n_clust)
                
                clusters = clusterer.fit_predict(scaled_features)
                sil_score = silhouette_score(scaled_features, clusters)
                if sil_score > best_score:
                    best_score = sil_score
                    best_n_clusters = n_clust
            n_clusters = best_n_clusters

        # Final clustering with chosen parameters
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'agglomerative':
            from sklearn.cluster import AgglomerativeClustering
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)

        final_clusters = clusterer.fit_predict(scaled_features)
        final_sil_score = silhouette_score(scaled_features, final_clusters)

        return final_clusters, n_clusters, final_sil_score

    def perform_pca_rankings(self, rankings_df, n_components=2, scaling=True):
        """Performs PCA on the ranking values with configurable options."""
        if rankings_df.empty or len(rankings_df) < 2:
            raise ValueError("PCA requer pelo menos 2 genótipos.")
        
        # Select only ranking columns
        rank_features = rankings_df.filter(regex='^Rank_')
        if rank_features.empty:
            raise ValueError("Não há colunas de ranking para PCA.")
            
        if scaling:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(rank_features)
        else:
            scaled_features = rank_features.values
            
        # Adjust n_components to available features
        max_components = min(n_components, rank_features.shape[1], rank_features.shape[0])
        pca = PCA(n_components=max_components)
        principal_components = pca.fit_transform(scaled_features)
        
        # Create column names dynamically
        pc_columns = [f'PC{i+1}' for i in range(max_components)]
        pca_df = pd.DataFrame(data=principal_components, columns=pc_columns)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        return pca_df, pca.explained_variance_ratio_, loadings, rank_features.columns, max_components

    def generate_dendrogram_linkage_rankings(self, rankings_df, method='ward', metric='euclidean'):
        """Generates the linkage matrix for a dendrogram using ranking values."""
        if rankings_df.empty or len(rankings_df) < 2:
            raise ValueError("Dendrograma requer pelo menos 2 genótipos.")
        
        # Select only ranking columns
        rank_features = rankings_df.filter(regex='^Rank_')
        if rank_features.empty:
            raise ValueError("Não há colunas de ranking para dendrograma.")
            
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(rank_features)
        return linkage(scaled_features, method=method, metric=metric)

    def calculate_descriptive_statistics(self, data):
        """Calculate descriptive statistics for numerical columns."""
        if data.empty:
            return pd.DataFrame()
            
        numeric_data = data.select_dtypes(include=np.number)
        if numeric_data.empty:
            return pd.DataFrame()
        
        stats = pd.DataFrame()
        stats['Média'] = numeric_data.mean()
        stats['Mediana'] = numeric_data.median()
        stats['Desvio Padrão'] = numeric_data.std()
        stats['Mínimo'] = numeric_data.min()
        stats['Máximo'] = numeric_data.max()
        stats['Q1'] = numeric_data.quantile(0.25)
        stats['Q3'] = numeric_data.quantile(0.75)
        stats['CV (%)'] = (numeric_data.std() / numeric_data.mean() * 100).round(2)
        stats['Assimetria'] = numeric_data.skew()
        stats['Curtose'] = numeric_data.kurtosis()
        
        return stats.round(4)
    
    def perform_normality_tests(self, data):
        """Perform comprehensive normality tests (Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling, D'Agostino-Pearson, Jarque-Bera) for numerical columns."""
        from scipy.stats import shapiro, kstest, anderson, normaltest
        from statsmodels.stats.stattools import jarque_bera
        import scipy.stats as stats
        
        if data.empty:
            return pd.DataFrame()
            
        numeric_data = data.select_dtypes(include=np.number)
        if numeric_data.empty:
            return pd.DataFrame()
        
        results = []
        for column in numeric_data.columns:
            values = numeric_data[column].dropna()
            n_values = len(values)
            
            if n_values < 3:
                # Add informative message for insufficient data
                results.append({
                    'Variável': column,
                    'Teste': 'INFORMAÇÃO',
                    'Estatística': 'N/A',
                    'p-valor': 'N/A',
                    'Interpretação': f'Dados insuficientes (n={n_values}). Mínimo: 3 observações.'
                })
                continue
                
            # Shapiro-Wilk test (for samples 3-5000)
            if 3 <= n_values <= 5000:
                try:
                    sw_stat, sw_p = shapiro(values)
                    if not (np.isnan(sw_stat) or np.isnan(sw_p)):
                        results.append({
                            'Variável': column,
                            'Teste': 'Shapiro-Wilk',
                            'Estatística': round(sw_stat, 4),
                            'p-valor': round(sw_p, 6),
                            'Interpretação': 'Normal' if sw_p > 0.05 else 'Não Normal'
                        })
                except Exception:
                    pass
            elif n_values > 5000:
                results.append({
                    'Variável': column,
                    'Teste': 'Shapiro-Wilk',
                    'Estatística': 'N/A',
                    'p-valor': 'N/A',
                    'Interpretação': f'Não aplicável (n={n_values}). Máximo: 5000 observações.'
                })
                    
            # Kolmogorov-Smirnov test (requires normalization, min 8 samples)
            if n_values >= 8:
                try:
                    # Normalize data
                    if values.std() > 0:  # Avoid division by zero
                        values_normalized = (values - values.mean()) / values.std()
                        ks_stat, ks_p = kstest(values_normalized, 'norm')
                        if not (np.isnan(ks_stat) or np.isnan(ks_p)):
                            results.append({
                                'Variável': column,
                                'Teste': 'Kolmogorov-Smirnov',
                                'Estatística': round(ks_stat, 4),
                                'p-valor': round(ks_p, 6),
                                'Interpretação': 'Normal' if ks_p > 0.05 else 'Não Normal'
                            })
                except Exception:
                    pass
            elif 3 <= n_values < 8:
                results.append({
                    'Variável': column,
                    'Teste': 'Kolmogorov-Smirnov',
                    'Estatística': 'N/A',
                    'p-valor': 'N/A',
                    'Interpretação': f'Não aplicável (n={n_values}). Mínimo: 8 observações.'
                })
                    
            # Anderson-Darling test (min 8 samples)
            if n_values >= 8:
                try:
                    ad_result = anderson(values, 'norm')
                    if not np.isnan(ad_result.statistic):
                        # Get critical value for 5% significance level (index 2)
                        critical_5 = ad_result.critical_values[2] if len(ad_result.critical_values) > 2 else None
                        if critical_5 is not None and not np.isnan(critical_5):
                            interpretation = 'Não Normal' if ad_result.statistic > critical_5 else 'Normal'
                            results.append({
                                'Variável': column,
                                'Teste': 'Anderson-Darling',
                                'Estatística': round(ad_result.statistic, 4),
                                'Valor Crítico (5%)': round(critical_5, 4),
                                'Interpretação': interpretation
                            })
                        else:
                            results.append({
                                'Variável': column,
                                'Teste': 'Anderson-Darling',
                                'Estatística': round(ad_result.statistic, 4),
                                'Valor Crítico (5%)': 'N/A',
                                'Interpretação': 'Inconclusivo - valores críticos não disponíveis'
                            })
                except Exception:
                    pass
            elif 3 <= n_values < 8:
                results.append({
                    'Variável': column,
                    'Teste': 'Anderson-Darling',
                    'Estatística': 'N/A',
                    'Valor Crítico (5%)': 'N/A',
                    'Interpretação': f'Não aplicável (n={n_values}). Mínimo: 8 observações.'
                })
                    
            # Jarque-Bera test (requires at least 8 observations, but more reliable with 20+)
            if n_values >= 8:
                try:
                    jb_stat, jb_p, _, _ = jarque_bera(values)
                    if not (np.isnan(jb_stat) or np.isnan(jb_p)):
                        note = " (mais confiável com n≥20)" if n_values < 20 else ""
                        results.append({
                            'Variável': column,
                            'Teste': 'Jarque-Bera',
                            'Estatística': round(jb_stat, 4),
                            'p-valor': round(jb_p, 6),
                            'Interpretação': ('Normal' if jb_p > 0.05 else 'Não Normal') + note
                        })
                except Exception:
                    pass
            elif 3 <= n_values < 8:
                results.append({
                    'Variável': column,
                    'Teste': 'Jarque-Bera',
                    'Estatística': 'N/A',
                    'p-valor': 'N/A',
                    'Interpretação': f'Não aplicável (n={n_values}). Mínimo: 8 observações.'
                })
                    
            # D'Agostino-Pearson test (requires at least 20 observations)
            if n_values >= 20:
                try:
                    dp_stat, dp_p = normaltest(values)
                    if not (np.isnan(dp_stat) or np.isnan(dp_p)):
                        results.append({
                            'Variável': column,
                            'Teste': "D'Agostino-Pearson",
                            'Estatística': round(dp_stat, 4),
                            'p-valor': round(dp_p, 6),
                            'Interpretação': 'Normal' if dp_p > 0.05 else 'Não Normal'
                        })
                except Exception:
                    pass
            elif 3 <= n_values < 20:
                results.append({
                    'Variável': column,
                    'Teste': "D'Agostino-Pearson",
                    'Estatística': 'N/A',
                    'p-valor': 'N/A',
                    'Interpretação': f'Não aplicável (n={n_values}). Mínimo: 20 observações.'
                })
        
        return pd.DataFrame(results)

    def calculate_mfv(self, indices):
        """Calculates the Membership Function Value (MFV) for each index."""
        if indices.empty:
            return pd.DataFrame()

        mfv_df = indices[['Genotipo']].copy()
        numeric_indices = indices.select_dtypes(include=np.number).drop(['Cor', 'Cluster'], axis=1, errors='ignore')

        # Indices where higher is better
        higher_is_better = ['Yp', 'Ys', 'MP', 'GMP', 'HMS', 'STI', 'YI', 'YSI', 'RSI']
        # Indices where lower is better
        lower_is_better = ['RP', 'TOL', 'SSI']

        for col in numeric_indices.columns:
            if col not in higher_is_better and col not in lower_is_better:
                continue

            max_val = numeric_indices[col].max()
            min_val = numeric_indices[col].min()
            
            # Avoid division by zero if all values are the same
            if (max_val - min_val) == 0:
                mfv_df[f'MFV_{col}'] = 1.0 # All genotypes are equally good
                continue

            if col in higher_is_better:
                mfv_df[f'MFV_{col}'] = (numeric_indices[col] - min_val) / (max_val - min_val)
            elif col in lower_is_better:
                mfv_df[f'MFV_{col}'] = 1 - ((numeric_indices[col] - min_val) / (max_val - min_val))

        # Calculate D value (average MFV) - exclude rank columns from calculation
        mfv_cols = [col for col in mfv_df.columns if col.startswith('MFV_') and not col.endswith('_Rank')]
        mfv_df['D_value'] = mfv_df[mfv_cols].mean(axis=1)

        # Rank based on D value (higher is better)
        mfv_df['MFV_Rank'] = mfv_df['D_value'].rank(method='min', ascending=False).astype(int)
        
        # Classification based on average MFV (Xa) and standard deviation (S.D.)
        xa = mfv_df['D_value'].mean()  # Average MFV value
        sd = mfv_df['D_value'].std()   # Standard deviation
        
        def classify_tolerance(d_value):
            """Classify genotype tolerance based on D value."""
            if d_value >= xa + 1.64 * sd:
                return "Altamente Tolerante"
            elif xa + sd <= d_value < xa + 1.64 * sd:
                return "Tolerante"
            elif xa - sd <= d_value < xa + sd:
                return "Moderadamente Tolerante"
            elif xa - 1.64 * sd <= d_value < xa - sd:
                return "Sensível"
            else:  # d_value < xa - 1.64 * sd
                return "Altamente Sensível"
        
        mfv_df['Classe_Tolerância'] = mfv_df['D_value'].apply(classify_tolerance)
        
        return mfv_df.sort_values(by='MFV_Rank')

class DataManager:
    """Handles file I/O and data validation."""

    def import_file(self, file_path):
        """
        Imports data from a CSV or Excel file.
        Args:
            file_path (str): The path to the file.
        Returns:
            pd.DataFrame: The imported data.
        Raises:
            ValueError: If the file format is not supported or file is empty.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"O arquivo não foi encontrado: {file_path}")

        try:
            if file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path, sheet_name=0)
            elif file_path.endswith(('.csv', '.txt')):
                # Use a robust CSV reading method
                df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8-sig')
            else:
                raise ValueError(f"Formato de arquivo não suportado: {os.path.basename(file_path)}")

            if df.empty:
                raise ValueError("O arquivo importado está vazio.")

            return df
        except Exception as e:
            logging.error(f"Error importing file {file_path}: {e}")

            raise ValueError(f"Falha ao ler o arquivo. Verifique se o formato está correto e se não está corrompido.\n\nDetalhe: {e}")

    def validate_and_standardize(self, df):
        """
        Validates the structure of the imported DataFrame and standardizes column names.
        It expects 2 or 3 columns: (Genotipo, Yp, Ys) or (Yp, Ys).
        Args:
            df (pd.DataFrame): The raw imported DataFrame.
        Returns:
            tuple: A tuple containing (standardized_DataFrame, original_column_names).
        Raises:
            ValueError: If the DataFrame has an invalid structure or bad data.
        """
        if df.shape[1] < 2:
            raise ValueError(f"O arquivo deve ter 2 ou 3 colunas, mas foram encontradas {df.shape[1]}.")

        # Drop rows where all elements are NaN
        df.dropna(how='all', inplace=True)
        if df.empty:
            raise ValueError("O arquivo não contém dados válidos após remover linhas vazias.")

        data = df.iloc[:, :3].copy()
        original_columns = list(data.columns)

        if data.shape[1] == 2:
            data.columns = ['Yp', 'Ys']
            data['Genotipo'] = [f'Genótipo_{i+1}' for i in range(len(data))]
            # Reorder for consistency
            data = data[['Genotipo', 'Yp', 'Ys']]
            display_columns = ['Genótipo (Automático)', original_columns[0], original_columns[1]]
        else:
            data.columns = ['Genotipo', 'Yp', 'Ys']
            display_columns = original_columns

        # Convert to numeric, coercing errors to NaN
        data['Yp'] = pd.to_numeric(data['Yp'], errors='coerce')
        data['Ys'] = pd.to_numeric(data['Ys'], errors='coerce')

        # Check for non-numeric data issues
        if data['Yp'].isnull().any() or data['Ys'].isnull().any():
            invalid_rows = data[data['Yp'].isnull() | data['Ys'].isnull()].index.tolist()
            linhas = ', '.join(str(r + 2) for r in invalid_rows)
            raise ValueError(f"Dados não numéricos encontrados nas colunas de produtividade.\nPor favor, verifique as linhas: {linhas}.")

        data.dropna(subset=['Yp', 'Ys'], inplace=True)
        if data.empty:
            raise ValueError("Nenhum dado válido restante após a limpeza. Verifique o conteúdo do arquivo.")

        return data[['Genotipo', 'Yp', 'Ys']], display_columns

class Worker(QThread):
    """A worker thread for running long-running tasks without freezing the UI."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """Executes the function and emits signals on completion or error."""
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            logging.error(f"Error in worker thread for function {self.func.__name__}: {e}", exc_info=True)
            self.error.emit(str(e))

class SettingsDialog(QDialog):
    """A dialog for configuring application settings, like classification limits."""
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle("Configurações de Classificação")
        self.settings = settings
        self.layout = QFormLayout(self)

        self.tolerant_spin = QSpinBox(); self.tolerant_spin.setRange(0, 100)
        self.mod_tolerant_spin = QSpinBox(); self.mod_tolerant_spin.setRange(0, 100)
        self.mod_susceptible_spin = QSpinBox(); self.mod_susceptible_spin.setRange(0, 100)

        self.layout.addRow("Tolerante (RP <=)", self.tolerant_spin)
        self.layout.addRow("Moderadamente Tolerante (RP <=)", self.mod_tolerant_spin)
        self.layout.addRow("Moderadamente Suscetível (RP <=)", self.mod_susceptible_spin)

        self.save_btn = QPushButton("Salvar"); self.cancel_btn = QPushButton("Cancelar")
        self.save_btn.clicked.connect(self.accept); self.cancel_btn.clicked.connect(self.reject)
        btn_layout = QHBoxLayout(); btn_layout.addWidget(self.save_btn); btn_layout.addWidget(self.cancel_btn)
        self.layout.addRow(btn_layout)
        self.load_settings()

    def load_settings(self):
        """Loads current settings into the spinboxes."""
        self.tolerant_spin.setValue(self.settings.get('tolerant', 20))
        self.mod_tolerant_spin.setValue(self.settings.get('mod_tolerant', 40))
        self.mod_susceptible_spin.setValue(self.settings.get('mod_susceptible', 60))

    def get_settings(self):
        """Returns the new settings from the spinboxes."""
        return {
            'tolerant': self.tolerant_spin.value(),
            'mod_tolerant': self.mod_tolerant_spin.value(),
            'mod_susceptible': self.mod_susceptible_spin.value()
        }

# --- Report Generator ---

class ReportGenerator:
    """
    Class responsible for generating text-based reports with tables.
    """
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.data = main_window.data
        self.indices = main_window.indices
        self.rankings_df = main_window.rankings_df
    
    def generate_indices_report(self):
        """Generate a comprehensive report of tolerance indices."""
        if self.indices.empty:
            return "Erro: Nenhum índice calculado. Execute o cálculo de índices primeiro."
        
        report = []
        report.append("=" * 80)
        report.append("RELATÓRIO DE ÍNDICES DE TOLERÂNCIA AO ESTRESSE")
        report.append("=" * 80)
        report.append(f"Data de geração: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        report.append("")
        
        # Informações gerais
        report.append("1. INFORMAÇÕES GERAIS")
        report.append("-" * 40)
        report.append(f"Número de genótipos analisados: {len(self.indices)}")
        numeric_cols = self.indices.select_dtypes(include=[np.number]).columns
        report.append(f"Número de índices calculados: {len(numeric_cols)}")
        report.append("")
        
        # Estatísticas descritivas
        report.append("2. ESTATÍSTICAS DESCRITIVAS DOS ÍNDICES")
        report.append("-" * 40)
        stats_df = self.indices[numeric_cols].describe()
        report.append(self._dataframe_to_text(stats_df, "Estatísticas"))
        report.append("")
        
        # Tabela completa de índices
        report.append("3. TABELA COMPLETA DE ÍNDICES")
        report.append("-" * 40)
        report.append(self._dataframe_to_text(self.indices, "Índices por Genótipo"))
        report.append("")
        
        # Rankings se disponível
        if not self.rankings_df.empty:
            report.append("4. RANKINGS DE TOLERÂNCIA")
            report.append("-" * 40)
            report.append(self._dataframe_to_text(self.rankings_df, "Rankings"))
            report.append("")
        
        # Classificação por tolerância
        if 'STI' in self.indices.columns:
            report.append("5. CLASSIFICAÇÃO POR TOLERÂNCIA (BASEADA NO STI)")
            report.append("-" * 40)
            classification = self._classify_genotypes()
            for class_name, genotypes in classification.items():
                report.append(f"{class_name}: {len(genotypes)} genótipos")
                for genotype in genotypes:
                    report.append(f"  - {genotype}")
                report.append("")
        
        return "\n".join(report)
    
    def generate_statistics_report(self):
        """Generate a statistical analysis report."""
        if self.indices.empty:
            return "Erro: Nenhum índice calculado. Execute o cálculo de índices primeiro."
        
        report = []
        report.append("=" * 80)
        report.append("RELATÓRIO DE ANÁLISE ESTATÍSTICA")
        report.append("=" * 80)
        report.append(f"Data de geração: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        report.append("")
        
        numeric_data = self.indices.select_dtypes(include=[np.number])
        
        # Teste de normalidade
        report.append("1. TESTE DE NORMALIDADE (JARQUE-BERA)")
        report.append("-" * 40)
        for col in numeric_data.columns:
            try:
                stat, p_value = jarque_bera(numeric_data[col].dropna())
                is_normal = "SIM" if p_value > 0.05 else "NÃO"
                report.append(f"{col}:")
                report.append(f"  Estatística: {stat:.4f}")
                report.append(f"  P-valor: {p_value:.4f}")
                report.append(f"  Distribuição normal: {is_normal}")
                report.append("")
            except:
                report.append(f"{col}: Erro no teste de normalidade")
                report.append("")
        
        # Matriz de correlação
        report.append("2. MATRIZ DE CORRELAÇÃO DE PEARSON")
        report.append("-" * 40)
        corr_matrix = numeric_data.corr()
        report.append(self._dataframe_to_text(corr_matrix, "Correlações"))
        report.append("")
        
        # Correlações significativas
        report.append("3. CORRELAÇÕES SIGNIFICATIVAS (|r| > 0.7)")
        report.append("-" * 40)
        significant_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    significant_corrs.append(f"{corr_matrix.columns[i]} x {corr_matrix.columns[j]}: {corr_val:.3f}")
        
        if significant_corrs:
            for corr in significant_corrs:
                report.append(f"  {corr}")
        else:
            report.append("  Nenhuma correlação significativa encontrada.")
        report.append("")
        
        return "\n".join(report)
    
    def generate_clustering_report(self, clustering_result=None):
        """Generate a clustering analysis report."""
        if clustering_result is None:
            return "Erro: Nenhum resultado de clustering disponível. Execute a análise de clustering primeiro."
        
        report = []
        report.append("=" * 80)
        report.append("RELATÓRIO DE ANÁLISE DE CLUSTERING")
        report.append("=" * 80)
        report.append(f"Data de geração: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        report.append("")
        
        # Informações do clustering
        report.append("1. INFORMAÇÕES DO CLUSTERING")
        report.append("-" * 40)
        if 'method' in clustering_result:
            report.append(f"Método utilizado: {clustering_result['method']}")
        if 'n_clusters' in clustering_result:
            report.append(f"Número de clusters: {clustering_result['n_clusters']}")
        if 'silhouette_score' in clustering_result:
            report.append(f"Silhouette Score: {clustering_result['silhouette_score']:.4f}")
        report.append("")
        
        # Distribuição dos clusters
        if 'labels' in clustering_result:
            report.append("2. DISTRIBUIÇÃO DOS CLUSTERS")
            report.append("-" * 40)
            labels = clustering_result['labels']
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                cluster_indices = np.where(labels == label)[0]
                report.append(f"Cluster {label + 1}: {len(cluster_indices)} genótipos")
                
                if hasattr(self.indices, 'index'):
                    genotypes = self.indices.index[cluster_indices].tolist()
                else:
                    # Tentar usar nomes de genótipos reais se disponível
                    if 'Genótipo' in self.indices.columns:
                        genotypes = [self.indices.iloc[i]['Genótipo'] for i in cluster_indices if i < len(self.indices)]
                    else:
                        genotypes = [f"Genótipo {i+1}" for i in cluster_indices]
                
                for genotype in genotypes:
                    report.append(f"  - {genotype}")
                report.append("")
        
        # Características dos clusters
        if 'cluster_stats' in clustering_result:
            report.append("3. CARACTERÍSTICAS DOS CLUSTERS")
            report.append("-" * 40)
            cluster_stats = clustering_result['cluster_stats']
            report.append(self._dataframe_to_text(cluster_stats, "Estatísticas por Cluster"))
            report.append("")
        
        return "\n".join(report)
    
    def _dataframe_to_text(self, df, title=""):
        """Convert DataFrame to formatted text table."""
        lines = []
        if title:
            lines.append(f"{title}:")
            lines.append("")
        
        # Header
        header = " | ".join([f"{col:>12}" for col in df.columns])
        lines.append(header)
        lines.append("-" * len(header))
        
        # Data rows
        for idx, row in df.iterrows():
            row_data = []
            for val in row:
                if isinstance(val, (int, float)) and not pd.isna(val):
                    if abs(val) < 0.001 or abs(val) > 1000:
                        row_data.append(f"{val:>12.2e}")
                    else:
                        row_data.append(f"{val:>12.3f}")
                else:
                    row_data.append(f"{str(val):>12}")
            lines.append(" | ".join(row_data))
        
        return "\n".join(lines)
    
    def _classify_genotypes(self):
        """Classify genotypes based on tolerance levels."""
        if 'STI' not in self.indices.columns:
            return {}
        
        sti_values = self.indices['STI']
        q25 = sti_values.quantile(0.25)
        q75 = sti_values.quantile(0.75)
        
        classification = {
            "Altamente Tolerantes (Q4)": [],
            "Moderadamente Tolerantes (Q3)": [],
            "Moderadamente Susceptíveis (Q2)": [],
            "Altamente Susceptíveis (Q1)": []
        }
        
        for idx, row in self.indices.iterrows():
            genotype = row.get('Genótipo', f'Genótipo {idx+1}')
            sti_val = row['STI']
            
            if sti_val >= q75:
                classification["Altamente Tolerantes (Q4)"].append(f"{genotype} (STI: {sti_val:.3f})")
            elif sti_val >= q25:
                classification["Moderadamente Tolerantes (Q3)"].append(f"{genotype} (STI: {sti_val:.3f})")
            else:
                classification["Altamente Susceptíveis (Q1)"].append(f"{genotype} (STI: {sti_val:.3f})")
        
        return classification
    
    def save_report_to_file(self, report_content, filename):
        """Save report content to a text file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            return True
        except Exception as e:
            return (False, str(e))

# --- Main Application Window ---

class MainWindow(QMainWindow):
    """
    The main application window, containing the UI and event handling logic.
    """
    def __init__(self):
        try:
            super().__init__()
            
            # Initialize settings (after QApplication exists)  
            self.settings_handler = None
            self.classification_limits = {'tolerant': 20.0, 'mod_tolerant': 40.0, 'mod_susceptible': 60.0}
            self.language_code = LANGUAGE_CODE

            self.setWindowTitle(GetFromLanguage('MainWindowTitle', 0))
            # self.setGeometry(100, 100, 1300, 850)  # Comentado para permitir maximização
            try:
                self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
            except Exception:
                pass

            # Initialize data holders and analysis objects
            self.data = pd.DataFrame()
            self.indices = pd.DataFrame()
            self.rankings_df = pd.DataFrame()
            self.original_column_names = [GetFromLanguage("Headers", i) for i in range(3)]
            self.current_graph_fig = None
            self.current_canvas = None
            self.last_click_time = 0
            try:
                self.analysis = IpasticAnalysis()
                self.enhanced_analysis = EnhancedAnalyses() if ENHANCED_AVAILABLE else None
            except Exception as e:
                QMessageBox.critical(self, "Erro ao inicializar análise", f"Erro: {str(e)}")
                self.analysis = None
                self.enhanced_analysis = None
            try:
                self.data_manager = DataManager()
            except Exception as e:
                QMessageBox.critical(self, "Erro ao inicializar DataManager", f"Erro: {str(e)}")
                self.data_manager = None
            
            # Initialize report generator
            self.report_generator = ReportGenerator(self)
            self.worker = None

            self.init_settings() # Initialize settings and language before creating the UI
            self.init_graph_controls()
            self.init_ui()
            self.update_ui_state() # Set initial state of widgets
            self.setAcceptDrops(True) # Enable drag and drop
        except Exception as e:
            QMessageBox.critical(None, "Erro na Inicialização", f"Erro ao inicializar a janela principal:\n{str(e)}")

    def init_settings(self):
        """Initialize settings after QApplication is available."""
        try:
            self.settings_handler = QSettings("Plant Stress Lab", "iPASTIC")
            # Load settings if available
            if self.settings_handler:
                saved_limits = self.settings_handler.value("classification_limits", None)
                if saved_limits:
                    self.classification_limits = saved_limits
        except Exception as e:
            print(f"Warning: Could not initialize settings: {e}")
            self.settings_handler = None

    def init_ui(self):
        """Initializes all UI components."""
        self.create_actions()
        self.create_menu()
        self.create_toolbar()

        self.central_widget = QTabWidget()
        self.setCentralWidget(self.central_widget)

        # Initialize advanced config widgets before creating tabs that use them
        self.pca_components_spin = QSpinBox()
        self.pca_components_spin.setRange(2, 10)
        self.pca_components_spin.setValue(2)
        
        # Novos controles PCA avançados
        self.pca_component_x_combo = QComboBox()
        self.pca_component_y_combo = QComboBox()
        self.pca_matrix_combo = QComboBox()
        self.pca_matrix_combo.addItems(['Correlação', 'Covariância'])
        self.pca_matrix_combo.setCurrentText('Correlação')
        self.pca_show_variance_check = QCheckBox("Mostrar percentuais")
        self.pca_show_variance_check.setChecked(True)
        
        self.pca_show_vectors_check = QCheckBox("Mostrar vetores de carregamento")
        self.pca_show_vectors_check.setChecked(True)
        
        # Inicializar combos de componentes
        for i in range(1, 11):  # PC1 até PC10
            self.pca_component_x_combo.addItem(f"PC{i}")
            self.pca_component_y_combo.addItem(f"PC{i}")
        
        # Definir valores padrão
        self.pca_component_x_combo.setCurrentText("PC1")
        self.pca_component_y_combo.setCurrentText("PC2")
        self.dendro_method_combo = QComboBox()
        self.dendro_method_combo.addItems(['ward', 'single', 'complete', 'average'])
        self.dendro_metric_combo = QComboBox()
        self.dendro_metric_combo.addItems(['euclidean', 'cityblock', 'cosine', 'correlation'])
        
        # Controles para divisão em grupos
        self.dendro_groups_spin = QSpinBox()
        self.dendro_groups_spin.setMinimum(1)
        self.dendro_groups_spin.setMaximum(10)
        self.dendro_groups_spin.setValue(1)  # Padrão: 1 grupo (tudo preto) 
        
        self.dendro_color_groups_check = QCheckBox("Colorir grupos")
        self.dendro_color_groups_check.setChecked(False)  # Padrão: desabilitado (preto)
        self.heatmap_method_combo = QComboBox()
        self.heatmap_method_combo.addItems(['pearson', 'spearman', 'kendall'])
        self.heatmap_values_check = QCheckBox("Mostrar valores numéricos")
        self.heatmap_values_check.setChecked(True)

        # Create tabs
        self.tab_input = self.create_input_tab()
        self.tab_indices = self.create_indices_tab()
        self.tab_rankings = self.create_rankings_tab()
        self.tab_statistics = self.create_statistics_tab()
        self.tab_graphics = self.create_graphics_tab()
        self.tab_pca_results = self.create_pca_results_tab()
        self.tab_reports = self.create_reports_tab()
        self.tab_regression = self.create_regression_tab()
        self.tab_help = self.create_help_tab()
        
        
        self.central_widget.addTab(self.tab_input, GetFromLanguage("Tabs",0))
        self.central_widget.addTab(self.tab_indices, GetFromLanguage("Tabs", 1))
        self.central_widget.addTab(self.tab_rankings, GetFromLanguage("Tabs", 2))
        self.central_widget.addTab(self.tab_statistics, GetFromLanguage("Tabs", 3))
        self.central_widget.addTab(self.tab_graphics, GetFromLanguage("Tabs", 4))
        self.central_widget.addTab(self.tab_pca_results, GetFromLanguage("Tabs", 5))
        self.central_widget.addTab(self.tab_regression, GetFromLanguage("Tabs", 6))
        self.central_widget.addTab(self.tab_reports, GetFromLanguage("Tabs", 7))
        self.central_widget.addTab(self.tab_help, GetFromLanguage("Tabs", 8))

        # Shortcuts will be added after table_input is created

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

    #TODO:FIX DUPLICATA
    def on_table_item_changed(self, item):
        """
        Called when a cell in the input table is changed by the user.
        Reads the current data from the table, updates the internal DataFrame,
        and refreshes the UI state.
        """
        # Disconnect the signal to prevent recursive calls while we process the table
        try:
            self.table_input.itemChanged.disconnect(self.on_table_item_changed)
        except TypeError:
            pass  # Signal might have already been disconnected

        rows = self.table_input.rowCount()
        cols = self.table_input.columnCount()
        
        # Use original column names for the DataFrame
        headers = self.original_column_names if len(self.original_column_names) == cols else [f'Col{i+1}' for i in range(cols)]

        table_data = []
        for r in range(rows):
            row_items = [self.table_input.item(r, c) for c in range(cols)]
            # Only include rows that have at least one non-empty cell
            if any(item and item.text().strip() for item in row_items):
                table_data.append([item.text().strip() if item else '' for item in row_items])

        if not table_data:
            self.data = pd.DataFrame()
        else:
            # Create a DataFrame from the table content
            df = pd.DataFrame(table_data, columns=headers)
            
            # Basic validation and standardization similar to file import
            try:
                if df.shape[1] == 2:
                    df.columns = ['Yp', 'Ys']
                    df['Genotipo'] = [f'Genótipo_{i+1}' for i in range(len(df))]
                    df = df[['Genotipo', 'Yp', 'Ys']]
                else:
                    df.columns = ['Genotipo', 'Yp', 'Ys']

                # Coerce to numeric, but don't raise errors, just for the state check
                df['Yp'] = pd.to_numeric(df['Yp'], errors='coerce')
                df['Ys'] = pd.to_numeric(df['Ys'], errors='coerce')
                
                # Drop rows where essential data is missing after coercion
                df.dropna(subset=['Yp', 'Ys'], how='all', inplace=True)
                self.data = df
            except (ValueError, TypeError) as e:
                # In case of error (e.g., wrong column count), treat as no data
                self.data = pd.DataFrame()
                logging.warning(f"Could not parse table data for UI update: {e}")

        # Reconnect the signal for future edits
        self.table_input.itemChanged.connect(self.on_table_item_changed)
        
        # Update the UI based on whether we have valid data now
        self.update_ui_state()

        # Automatically start calculation if data is valid
        if self.calc_action.isEnabled():
            self.start_calculation()

    def create_actions(self):
        """Creates QAction objects for menus and toolbars."""
        self.import_action = QAction(self.style().standardIcon(QStyle.SP_DirIcon), GetFromLanguage("MenuFile", 1), self,
                                     shortcut=QKeySequence.Open, triggered=self.import_data)
        self.export_excel_action = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), GetFromLanguage("MenuFile", 2), self,
                                           shortcut="Ctrl+E", triggered=self.export_to_excel)
        self.clear_action = QAction(self.style().standardIcon(QStyle.SP_TrashIcon), GetFromLanguage("MenuFile", 3), self,
                                    shortcut="Ctrl+D", triggered=self.clear_all_data)
        self.exit_action = QAction(GetFromLanguage("MenuFile", 4), self, shortcut="Ctrl+Q", triggered=self.close)

        self.calc_action = QAction(self.style().standardIcon(QStyle.SP_MediaPlay), GetFromLanguage("MenuTools", 1), self,
                                   triggered=self.start_calculation)
        self.settings_action = QAction(self.style().standardIcon(QStyle.SP_FileDialogDetailedView), GetFromLanguage("MenuTools", 2), self,
                                       triggered=self.open_settings)
        self.change_language = QAction(self.style().standardIcon(QStyle.SP_FileDialogDetailedView), GetFromLanguage("MenuFile", 5), self,
                                       triggered=self.change_language)
        
        # Report actions
        self.indices_report_action = QAction(self.style().standardIcon(QStyle.SP_FileDialogDetailedView), GetFromLanguage("MenuReports", 1), self,
                                            triggered=self.generate_indices_report)
        self.statistics_report_action = QAction(self.style().standardIcon(QStyle.SP_FileDialogDetailedView), GetFromLanguage("MenuReports", 2), self,
                                               triggered=self.generate_statistics_report)
        self.clustering_report_action = QAction(self.style().standardIcon(QStyle.SP_FileDialogDetailedView), GetFromLanguage("MenuReports", 3), self,
                                               triggered=self.generate_clustering_report)

    def create_menu(self):
        """Creates the main menu bar."""
        menubar = self.menuBar()
        file_menu = menubar.addMenu(GetFromLanguage("MenuFile", 0))
        file_menu.addAction(self.import_action)
        file_menu.addAction(self.export_excel_action)
        file_menu.addSeparator()
        file_menu.addAction(self.clear_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        file_menu.addSeparator()
        file_menu.addAction(self.change_language)

        tools_menu = menubar.addMenu(GetFromLanguage("MenuTools", 0))
        tools_menu.addAction(self.calc_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.settings_action)
        
        reports_menu = menubar.addMenu(GetFromLanguage("MenuReports", 0))
        reports_menu.addAction(self.indices_report_action)
        reports_menu.addAction(self.statistics_report_action)
        reports_menu.addAction(self.clustering_report_action)

    def create_toolbar(self):
        """Creates the main toolbar."""
        toolbar = QToolBar(GetFromLanguage("Toolbar", 0))
        toolbar.setIconSize(QSize(32, 32))  # Larger icons
        self.addToolBar(toolbar)
        toolbar.addAction(self.import_action)
        toolbar.addAction(self.clear_action)
        toolbar.addSeparator()
        toolbar.addAction(self.calc_action)
        toolbar.addSeparator()
        toolbar.addAction(self.export_excel_action)
        toolbar.addSeparator()
        toolbar.addAction(self.change_language)

    #TODO:FIX DUPLICATA
    def get_professional_group_style(self):
        """Retorna um estilo CSS profissional para QGroupBox."""
        return """
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #333;
                border: 1px solid #D0D0D0;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                background-color: #F0F0F0;
                border-radius: 4px;
            }
        """

    def get_professional_button_style(self):
        """Retorna um estilo CSS profissional para botões."""
        return """
            QPushButton {
                background-color: #4CAF50; /* Verde */
                color: white;
                padding: 10px 15px;
                font-weight: bold;
                border-radius: 5px;
                border: 1px solid #388E3C;
            }
            QPushButton:hover {
                background-color: #45a049; /* Verde mais escuro */
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
                border-color: #999999;
            }
        """

    def toggle_report_sections(self, state):
        """Marca ou desmarca todas as checkboxes de seção do relatório."""
        check_state = Qt.Checked if state == Qt.Checked else Qt.Unchecked
        
        # Itera sobre todas as checkboxes que são filhas dos group boxes
        for group_box in [self.basic_options_group, self.regression_options_group, self.pca_options_group]:
            if hasattr(self, 'basic_options_group') and self.basic_options_group is not None:
                checkboxes = group_box.findChildren(QCheckBox)
                for checkbox in checkboxes:
                    # Não alterar o estado do "Selecionar Todos"
                    if "Todos" not in checkbox.text():
                        checkbox.setCheckState(check_state)

    def create_reports_tab(self):
        """Cria a aba de relatórios com layout e estilo aprimorados."""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # --- Título e Descrição ---
        title_label = QLabel(GetFromLanguage("ReportsTab", 0))
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #333;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        info_label = QLabel(GetFromLanguage("ReportsTab", 1))
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("font-size: 14px; color: #666; margin-bottom: 20px;")
        main_layout.addWidget(info_label)

        # --- Layout Principal (Duas Colunas) ---
        content_layout = QHBoxLayout()
        left_column = QVBoxLayout()
        right_column = QVBoxLayout()
        
        content_layout.addLayout(left_column, 2)  # Coluna de opções com 2/3 do espaço
        content_layout.addLayout(right_column, 1) # Coluna de geração com 1/3 do espaço

        # --- Coluna Esquerda: Opções de Conteúdo ---
        options_container = QGroupBox(GetFromLanguage("ReportsTab", 2))
        options_container.setStyleSheet(self.get_professional_group_style())
        options_layout = QVBoxLayout(options_container)
        
        # Botões de Seleção Rápida
        selection_buttons_layout = QHBoxLayout()
        select_all_button = QPushButton(GetFromLanguage("ReportsTab", 3))
        deselect_all_button = QPushButton(GetFromLanguage("ReportsTab", 4))
        select_all_button.setStyleSheet("padding: 5px;")
        deselect_all_button.setStyleSheet("padding: 5px;")
        
        selection_buttons_layout.addWidget(select_all_button)
        selection_buttons_layout.addWidget(deselect_all_button)
        selection_buttons_layout.addStretch()
        options_layout.addLayout(selection_buttons_layout)

        # Agrupamento das checkboxes
        self.basic_options_group = QGroupBox(GetFromLanguage("ReportsTab", 5))
        self.regression_options_group = QGroupBox(GetFromLanguage("ReportsTab", 6))
        self.pca_options_group = QGroupBox(GetFromLanguage("ReportsTab", 7))

        # Layouts para os grupos
        basic_layout = QVBoxLayout(self.basic_options_group)
        regression_layout = QVBoxLayout(self.regression_options_group)
        pca_layout = QVBoxLayout(self.pca_options_group)

        # Checkboxes de Análises Básicas
        self.include_descriptive = QCheckBox(GetFromLanguage("ReportsTab", 8))
        self.include_correlation = QCheckBox(GetFromLanguage("ReportsTab", 9))
        self.include_normality = QCheckBox(GetFromLanguage("ReportsTab", 10))
        self.include_outliers = QCheckBox(GetFromLanguage("ReportsTab", 11))
        self.include_clustering = QCheckBox(GetFromLanguage("ReportsTab", 12))
        
        basic_checkboxes = [self.include_descriptive, self.include_correlation, self.include_normality, self.include_outliers, self.include_clustering]
        for cb in basic_checkboxes:
            cb.setChecked(True)
            basic_layout.addWidget(cb)

        # Checkboxes de Regressão
        self.include_regression_summary = QCheckBox(GetFromLanguage("ReportsTab", 13))
        self.include_regression_equation = QCheckBox(GetFromLanguage("ReportsTab", 14))
        self.include_regression_coefficients = QCheckBox(GetFromLanguage("ReportsTab", 15))
        self.include_regression_diagnostics = QCheckBox(GetFromLanguage("ReportsTab", 16))
        self.include_regression_residuals = QCheckBox(GetFromLanguage("ReportsTab", 17))
        self.include_regression_values = QCheckBox(GetFromLanguage("ReportsTab", 18))

        regression_checkboxes = [self.include_regression_summary, self.include_regression_equation, self.include_regression_coefficients, self.include_regression_diagnostics, self.include_regression_residuals, self.include_regression_values]
        for cb in regression_checkboxes:
            cb.setChecked(True)
            regression_layout.addWidget(cb)

        # Checkboxes de PCA
        self.include_pca_eigenvalues = QCheckBox(GetFromLanguage("ReportsTab", 19))
        self.include_pca_eigenvectors = QCheckBox(GetFromLanguage("ReportsTab", 20))
        self.include_pca_loadings = QCheckBox(GetFromLanguage("ReportsTab", 21))
        self.include_pca_coordinates = QCheckBox(GetFromLanguage("ReportsTab", 22))

        pca_checkboxes = [self.include_pca_eigenvalues, self.include_pca_eigenvectors, self.include_pca_loadings, self.include_pca_coordinates]
        for cb in pca_checkboxes:
            cb.setChecked(True)
            pca_layout.addWidget(cb)

        # Adicionar grupos ao layout de opções
        options_layout.addWidget(self.basic_options_group)
        options_layout.addWidget(self.regression_options_group)
        options_layout.addWidget(self.pca_options_group)
        
        left_column.addWidget(options_container)
        left_column.addStretch()

        # --- Coluna Direita: Geração do Relatório ---
        generation_container = QGroupBox(GetFromLanguage("ReportsTab", 23))
        generation_container.setStyleSheet(self.get_professional_group_style())
        generation_layout = QFormLayout(generation_container)
        generation_layout.setRowWrapPolicy(QFormLayout.WrapAllRows)

        # Formato de Saída
        self.report_format = QComboBox()
        self.report_format.addItems([GetFromLanguage("ReportsTab", 27)])
        self.report_format.setStyleSheet("padding: 5px;")
        generation_layout.addRow(QLabel(GetFromLanguage("ReportsTab", 24)), self.report_format)

        # Botão de Geração
        self.generate_report_btn = QPushButton(GetFromLanguage("ReportsTab", 25))
        self.generate_report_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.generate_report_btn.setStyleSheet(self.get_professional_button_style())
        self.generate_report_btn.clicked.connect(self.generate_custom_report)
        
        # Status do Relatório
        self.report_status_label = QLabel(GetFromLanguage("ReportsTab", 26))
        self.report_status_label.setWordWrap(True)
        self.report_status_label.setStyleSheet("font-size: 12px; color: #777; margin-top: 15px;")

        right_column.addWidget(generation_container)
        right_column.addSpacing(20)
        right_column.addWidget(self.generate_report_btn)
        right_column.addWidget(self.report_status_label)
        right_column.addStretch()

        # Conectar botões de seleção
        select_all_button.clicked.connect(lambda: self.toggle_report_sections(Qt.Checked))
        deselect_all_button.clicked.connect(lambda: self.toggle_report_sections(Qt.Unchecked))

        main_layout.addLayout(content_layout)
        return tab

    def generate_custom_report(self):
        """Gera relatório personalizado em Excel."""
        if self.indices.empty:
            QMessageBox.warning(self, GetFromLanguage("Dialogs", 0), GetFromLanguage("Dialogs", 1))
            return
            
        # Escolher arquivo de saída
        file_path, _ = QFileDialog.getSaveFileName(
            self, GetFromLanguage("Dialogs", 2), "relatorio_ipastic.xlsx", GetFromLanguage("Dialogs", 3))
        if file_path:
            self._generate_excel_report(file_path)
    
    def _generate_excel_report(self, file_path):
        """Gera relatório em Excel."""
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Dados de entrada
                self.data.to_excel(writer, sheet_name='1. Dados de Entrada', index=False)
                
                # Índices calculados
                self.indices.to_excel(writer, sheet_name='2. Índices Calculados', index=False)
                
                # Rankings
                if not self.rankings_df.empty:
                    self.rankings_df.to_excel(writer, sheet_name='3. Rankings', index=False)
                
                # Análises estatísticas
                if self.include_descriptive.isChecked():
                    try:
                        data_source = self.indices.select_dtypes(include=np.number)
                        desc_stats = self.analysis.calculate_descriptive_statistics(data_source)
                        if not desc_stats.empty:
                            desc_stats.to_excel(writer, sheet_name='4. Estatísticas Descritivas')
                    except Exception as e:
                        logging.warning(f"Failed to add descriptive stats: {e}")
                
                if self.include_correlation.isChecked():
                    try:
                        data_source = self.indices.select_dtypes(include=np.number)
                        corr_matrix = data_source.corr()
                        if not corr_matrix.empty:
                            corr_matrix.to_excel(writer, sheet_name='5. Matriz de Correlação')
                    except Exception as e:
                        logging.warning(f"Failed to add correlation matrix: {e}")
                
                if self.include_normality.isChecked():
                    try:
                        data_source = self.indices.select_dtypes(include=np.number)
                        norm_results = self.analysis.perform_normality_tests(data_source)
                        if not norm_results.empty:
                            norm_results.to_excel(writer, sheet_name='6. Testes de Normalidade', index=False)
                    except Exception as e:
                        logging.warning(f"Failed to add normality tests: {e}")
                
                if self.include_outliers.isChecked():
                    try:
                        outliers_results = self.analysis.detect_outliers(self.indices)
                        if not outliers_results.empty:
                            outliers_results.to_excel(writer, sheet_name='7. Detecção de Outliers', index=False)
                    except Exception as e:
                        logging.warning(f"Failed to add outliers: {e}")
                
                # Adicionar seções de regressão se disponíveis
                self._add_regression_sections_to_excel(writer)
                
                # Adicionar seções de PCA se disponíveis
                self._add_pca_sections_to_excel(writer)
                
            
            self.report_status_label.setText(f"✅ Relatório Excel gerado com sucesso: {file_path}")
            self.status_bar.showMessage("Relatório Excel gerado!", 5000)
            
        except Exception as e:
            QMessageBox.critical(self, "Erro ao Gerar Relatório", 
                               f"Erro ao gerar relatório Excel:\n{str(e)}")
            self.report_status_label.setText("❌ Erro ao gerar relatório Excel")
    
    def _add_regression_sections_to_excel(self, writer):
        """Adiciona seções de regressão ao relatório Excel."""
        try:
            import pandas as pd
            # Verificar se existem resultados de regressão
            print(f"DEBUG: Verificando resultados de regressão...")
            print(f"DEBUG: hasattr current_regression_results: {hasattr(self, 'current_regression_results')}")
            if hasattr(self, 'current_regression_results'):
                print(f"DEBUG: current_regression_results: {self.current_regression_results is not None}")
            
            if not hasattr(self, 'current_regression_results') or not self.current_regression_results:
                print("DEBUG: Não há resultados de regressão disponíveis")
                return
            
            print("DEBUG: Adicionando seções de regressão ao Excel...")
                
            results = self.current_regression_results
            model = results['model']
            
            # 1. Resumo do Modelo
            if self.include_regression_summary.isChecked():
                print("DEBUG: Criando resumo do modelo...")
                try:
                    summary_data = []
                    r2_pct = model.rsquared * 100
                    r2_adj_pct = model.rsquared_adj * 100
                    
                    summary_data.append(['Métrica', 'Valor'])
                    summary_data.append(['R² (R-squared)', f'{r2_pct:.4f}%'])
                    summary_data.append(['R² Ajustado (Adj R-squared)', f'{r2_adj_pct:.4f}%'])
                    summary_data.append(['Estatística F', f'{model.fvalue:.4f}'])
                    summary_data.append(['Valor-p F', f'{model.f_pvalue:.4f}'])
                    summary_data.append(['AIC', f'{model.aic:.4f}'])
                    summary_data.append(['BIC', f'{model.bic:.4f}'])
                    summary_data.append(['Log-Likelihood', f'{model.llf:.4f}'])
                    summary_data.append(['Número de Observações', f'{int(results["n_obs"])}'])
                    
                    df_summary = pd.DataFrame(summary_data[1:], columns=summary_data[0])
                    df_summary.to_excel(writer, sheet_name='Reg_1_Resumo', index=False)
                    print("DEBUG: Planilha de resumo criada com sucesso!")
                except Exception as e:
                    print(f"DEBUG: Erro ao criar resumo: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 2. Equação do Modelo
            if self.include_regression_equation.isChecked():
                print("DEBUG: Criando equação do modelo...")
                equation = results.get('equation', 'Equação não disponível')
                print(f"DEBUG: Equação obtida: {equation}")
                equation_data = [['Equação do Modelo'], [equation]]
                df_equation = pd.DataFrame(equation_data[1:], columns=equation_data[0])
                df_equation.to_excel(writer, sheet_name='Reg_2_Equacao', index=False)
                print("DEBUG: Planilha de equação criada com sucesso!")
            
            # 3. Coeficientes e Significância
            if self.include_regression_coefficients.isChecked():
                print("DEBUG: Criando coeficientes e significância...")
                try:
                    coef_data = []
                    coef_data.append(['Variável', 'Coeficiente', 'Erro Padrão', 'Estatística t', 'Valor-p', 'IC 95% Inferior', 'IC 95% Superior'])
                    
                    # Obter intervalos de confiança
                    conf_int = model.conf_int()
                    print(f"DEBUG: Intervalos de confiança: {conf_int}")
                    
                    # Obter nomes dos parâmetros
                    if hasattr(model.params, 'index'):
                        param_names = list(model.params.index)
                        params = model.params
                        bse = model.bse
                        tvalues = model.tvalues
                        pvalues = model.pvalues
                    else:
                        # Se params é numpy array, usar var_names dos resultados
                        param_names = results.get('var_names', [f'Var_{i}' for i in range(len(model.params))])
                        params = model.params
                        bse = model.bse if hasattr(model.bse, '__len__') else [model.bse] * len(params)
                        tvalues = model.tvalues if hasattr(model.tvalues, '__len__') else [model.tvalues] * len(params)
                        pvalues = model.pvalues if hasattr(model.pvalues, '__len__') else [model.pvalues] * len(params)
                    
                    print(f"DEBUG: param_names: {param_names}")
                    print(f"DEBUG: params type: {type(params)}")
                    
                    for i, param_name in enumerate(param_names):
                        if hasattr(params, 'iloc'):
                            coef = params.iloc[i] if i < len(params) else 0
                            std_err = bse.iloc[i] if i < len(bse) else 0
                            t_stat = tvalues.iloc[i] if i < len(tvalues) else 0
                            p_val = pvalues.iloc[i] if i < len(pvalues) else 0
                        else:
                            coef = params[i] if i < len(params) else 0
                            std_err = bse[i] if i < len(bse) else 0
                            t_stat = tvalues[i] if i < len(tvalues) else 0
                            p_val = pvalues[i] if i < len(pvalues) else 0
                        
                        # Intervalos de confiança
                        if hasattr(conf_int, 'iloc'):
                            ci_lower = conf_int.iloc[i, 0] if i < len(conf_int) else 0
                            ci_upper = conf_int.iloc[i, 1] if i < len(conf_int) else 0
                        else:
                            ci_lower = conf_int[i, 0] if i < len(conf_int) else 0
                            ci_upper = conf_int[i, 1] if i < len(conf_int) else 0
                        
                        coef_data.append([
                            param_name, 
                            f'{coef:.6f}',
                            f'{std_err:.6f}',
                            f'{t_stat:.6f}',
                            f'{p_val:.6f}',
                            f'{ci_lower:.6f}',
                            f'{ci_upper:.6f}'
                        ])
                    
                    df_coef = pd.DataFrame(coef_data[1:], columns=coef_data[0])
                    df_coef.to_excel(writer, sheet_name='Reg_3_Coeficientes', index=False)
                    print("DEBUG: Planilha de coeficientes criada com sucesso!")
                except Exception as e:
                    print(f"DEBUG: Erro ao criar coeficientes: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 4. Diagnósticos do Modelo
            if self.include_regression_diagnostics.isChecked():
                print("DEBUG: Criando diagnósticos do modelo...")
                try:
                    diag_data = []
                    diag_data.append(['Estatística/Teste', 'Valor', 'Interpretação'])
                    
                    # Estatísticas básicas do modelo
                    diag_data.append(['Número de observações', f'{int(results["n_obs"])}', 'Tamanho da amostra'])
                    diag_data.append(['Graus de liberdade (modelo)', f'{model.df_model:.1f}', 'Parâmetros estimados'])
                    diag_data.append(['Graus de liberdade (resíduos)', f'{model.df_resid:.1f}', 'Obs - parâmetros'])
                    diag_data.append(['R-quadrado', f'{model.rsquared:.6f}', 'Variação explicada'])
                    diag_data.append(['R-quadrado ajustado', f'{model.rsquared_adj:.6f}', 'R² ajustado pelo n° de variáveis'])
                    
                    # Calcular raiz do MSE
                    import numpy as np
                    mse_resid = model.mse_resid
                    rmse = np.sqrt(mse_resid)
                    diag_data.append(['Raiz do MSE', f'{rmse:.6f}', 'Erro padrão dos resíduos'])
                    
                    diag_data.append(['AIC', f'{model.aic:.4f}', 'Critério de informação de Akaike'])
                    diag_data.append(['BIC', f'{model.bic:.4f}', 'Critério de informação bayesiano'])
                    diag_data.append(['Log-Likelihood', f'{model.llf:.4f}', 'Log-verossimilhança'])
                    diag_data.append(['F-estatística', f'{model.fvalue:.4f}', 'Teste F do modelo'])
                    diag_data.append(['P-valor (F)', f'{model.f_pvalue:.4e}', 'Significância do modelo'])
                    diag_data.append(['Condição do modelo', f'{model.condition_number:.2f}', 'Multicolinearidade'])
                    
                    # Teste de homocedasticidade (Breusch-Pagan)
                    try:
                        from statsmodels.stats.diagnostic import het_breuschpagan
                        bp_lm, bp_lm_pval, bp_f, bp_f_pval = het_breuschpagan(model.resid, model.model.exog)
                        diag_data.append(['Breusch-Pagan (estat.)', f'{bp_lm:.4f}', 'Teste de homocedasticidade'])
                        diag_data.append(['Breusch-Pagan (p-valor)', f'{bp_lm_pval:.4e}', 'Homocedástico se > 0.05'])
                    except Exception as bp_e:
                        print(f"DEBUG: Erro no teste Breusch-Pagan: {bp_e}")
                        diag_data.append(['Breusch-Pagan (estat.)', 'Erro no cálculo', 'N/A'])
                        diag_data.append(['Breusch-Pagan (p-valor)', 'Erro no cálculo', 'N/A'])
                    
                    # Teste de normalidade (Jarque-Bera)
                    try:
                        from scipy import stats
                        jb_stat, jb_pval = stats.jarque_bera(model.resid)
                        diag_data.append(['Jarque-Bera (estat.)', f'{jb_stat:.4f}', 'Teste de normalidade'])
                        diag_data.append(['Jarque-Bera (p-valor)', f'{jb_pval:.4e}', 'Normal se > 0.05'])
                    except Exception as jb_e:
                        print(f"DEBUG: Erro no teste Jarque-Bera: {jb_e}")
                        # Fallback - calculate manually using scipy.stats
                        try:
                            import numpy as np
                            from scipy import stats
                            # Manual Jarque-Bera calculation
                            residuals = model.resid
                            n = len(residuals)
                            skewness = stats.skew(residuals)
                            kurtosis = stats.kurtosis(residuals)
                            jb_stat = (n/6) * (skewness**2 + (kurtosis**2)/4)
                            jb_pval = 1 - stats.chi2.cdf(jb_stat, 2)
                            diag_data.append(['Jarque-Bera (estat.)', f'{jb_stat:.4f}', 'Teste de normalidade'])
                            diag_data.append(['Jarque-Bera (p-valor)', f'{jb_pval:.4e}', 'Normal se > 0.05'])
                        except:
                            diag_data.append(['Jarque-Bera (estat.)', 'Erro no cálculo', 'N/A'])
                            diag_data.append(['Jarque-Bera (p-valor)', 'Erro no cálculo', 'N/A'])
                    
                    # Teste de homocedasticidade (Breusch-Pagan)
                    try:
                        from statsmodels.stats.diagnostic import het_breuschpagan
                        bp_lm, bp_lm_pval, bp_f, bp_f_pval = het_breuschpagan(model.resid, model.model.exog)
                        bp_interp = 'Homocedástico' if bp_lm_pval > 0.05 else 'Heterocedástico'
                        diag_data.append(['Breusch-Pagan (Homocedasticidade)', f'LM={bp_lm:.4f}, p={bp_lm_pval:.4f}', bp_interp])
                    except Exception as bp_e:
                        print(f"DEBUG: Erro no teste Breusch-Pagan: {bp_e}")
                        diag_data.append(['Breusch-Pagan (estat.)', 'Erro no cálculo', 'N/A'])
                        diag_data.append(['Breusch-Pagan (p-valor)', 'Erro no cálculo', 'N/A'])
                    
                    # Durbin-Watson (Autocorrelação)
                    try:
                        from statsmodels.stats.stattools import durbin_watson
                        dw_stat = durbin_watson(model.resid)
                        diag_data.append(['Durbin-Watson', f'{dw_stat:.4f}', 'Autocorrelação: normal se 1.5-2.5'])
                    except Exception as dw_e:
                        print(f"DEBUG: Erro no teste Durbin-Watson: {dw_e}")
                        diag_data.append(['Durbin-Watson', 'Erro no cálculo', 'N/A'])
                    
                    df_diag = pd.DataFrame(diag_data[1:], columns=diag_data[0])
                    df_diag.to_excel(writer, sheet_name='Reg_4_Diagnosticos', index=False)
                    print("DEBUG: Planilha de diagnósticos criada com sucesso!")
                except Exception as e:
                    print(f"DEBUG: Erro ao criar diagnósticos: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 5. Análise de Resíduos
            if self.include_regression_residuals.isChecked():
                print("DEBUG: Criando análise de resíduos...")
                try:
                    resid_data = []
                    residuals = model.resid
                    
                    resid_data.append(['Estatística', 'Valor'])
                    resid_data.append(['Média dos Resíduos', f'{residuals.mean():.6f}'])
                    resid_data.append(['Desvio Padrão dos Resíduos', f'{residuals.std():.6f}'])
                    resid_data.append(['Resíduo Mínimo', f'{residuals.min():.6f}'])
                    resid_data.append(['Resíduo Máximo', f'{residuals.max():.6f}'])
                    # Usar numpy para quantis se residuals for numpy array
                    import numpy as np
                    if hasattr(residuals, 'quantile'):
                        q25 = residuals.quantile(0.25)
                        median = residuals.median()
                        q75 = residuals.quantile(0.75)
                    else:
                        q25 = np.percentile(residuals, 25)
                        median = np.median(residuals)
                        q75 = np.percentile(residuals, 75)
                    
                    resid_data.append(['Quartil 25%', f'{q25:.6f}'])
                    resid_data.append(['Mediana', f'{median:.6f}'])
                    resid_data.append(['Quartil 75%', f'{q75:.6f}'])
                    
                    # Criar planilha com estatísticas descritivas
                    df_resid = pd.DataFrame(resid_data[1:], columns=resid_data[0])
                    df_resid.to_excel(writer, sheet_name='Reg_5_Residuos', index=False)
                    
                    # Adicionar também uma tabela detalhada de resíduos por genótipo
                    residuals_detail_data = []
                    residuals_detail_data.append(['Genótipo', 'Resíduo', 'Resíduo Padronizado', 'Resíduo Absoluto'])
                    
                    # Obter nomes dos genótipos usando a mesma lógica
                    genotype_col = results.get('genotype_col', 'Genótipo')
                    data_copy = results['data_copy']
                    
                    residuals_array = residuals.values if hasattr(residuals, 'values') else residuals
                    std_residuals = np.std(residuals_array)
                    
                    for i in range(len(residuals_array)):
                        # Usar nomes reais dos genótipos se disponível
                        if genotype_col and genotype_col in data_copy.columns and i < len(data_copy):
                            genotype = str(data_copy.iloc[i][genotype_col])
                        else:
                            # Fallback para numeração sequencial
                            genotype = f'{i+1}'
                        
                        resid_val = residuals_array[i]
                        resid_std = resid_val / std_residuals if std_residuals != 0 else 0
                        resid_abs = abs(resid_val)
                        
                        residuals_detail_data.append([
                            genotype,
                            f'{resid_val:.6f}',
                            f'{resid_std:.6f}',
                            f'{resid_abs:.6f}'
                        ])
                    
                    # Criar DataFrame e adicionar à planilha existente (na mesma aba, após as estatísticas)
                    worksheet = writer.sheets['Reg_5_Residuos']
                    
                    # Adicionar uma linha em branco
                    start_row = len(df_resid) + 3
                    worksheet.write(start_row, 0, "Resíduos por Genótipo")
                    
                    # Adicionar cabeçalhos da tabela detalhada
                    for col, header in enumerate(residuals_detail_data[0]):
                        worksheet.write(start_row + 2, col, header)
                    
                    # Adicionar dados da tabela detalhada
                    for row_idx, row_data in enumerate(residuals_detail_data[1:], start=start_row + 3):
                        for col_idx, value in enumerate(row_data):
                            worksheet.write(row_idx, col_idx, value)
                    
                    print("DEBUG: Planilha de resíduos com estatísticas e valores individuais criada com sucesso!")
                except Exception as e:
                    print(f"DEBUG: Erro ao criar análise de resíduos: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 6. Valores Observados vs Estimados
            if self.include_regression_values.isChecked():
                print("DEBUG: Criando valores observados vs estimados...")
                try:
                    y_var = results['y_var']
                    genotype_col = results.get('genotype_col', 'Genótipo')
                    print(f"DEBUG: y_var={y_var}, genotype_col={genotype_col}")
                    
                    # Obter dados originais
                    data_copy = results['data_copy']
                    y_observed = data_copy[y_var]
                    y_predicted = model.fittedvalues
                    residuals = model.resid
                    
                    print(f"DEBUG: Tamanhos - observed: {len(y_observed)}, predicted: {len(y_predicted)}, residuals: {len(residuals)}")
                    
                    values_data = []
                    values_data.append(['Genótipo', 'Valor Observado', 'Valor Estimado', 'Resíduo', 'Resíduo %', 'Resíduo Absoluto'])
                    
                    # Criar lista única de dados para evitar duplicação
                    data_rows = []
                    n_obs = len(y_observed)
                    
                    for i in range(n_obs):
                        # Usar nomes reais dos genótipos se disponível
                        if genotype_col and genotype_col in data_copy.columns and i < len(data_copy):
                            genotype = str(data_copy.iloc[i][genotype_col])
                        else:
                            # Fallback para numeração sequencial simples: 1, 2, 3, ..., 90
                            genotype = f'{i+1}'
                        
                        # Lidar com numpy arrays e pandas Series
                        if hasattr(y_observed, 'iloc'):
                            obs = y_observed.iloc[i]
                        else:
                            obs = y_observed[i]
                        
                        if hasattr(y_predicted, 'iloc'):
                            pred = y_predicted.iloc[i]
                        else:
                            pred = y_predicted[i]
                        
                        if hasattr(residuals, 'iloc'):
                            resid = residuals.iloc[i]
                        else:
                            resid = residuals[i]
                        
                        resid_pct = (resid / obs * 100) if obs != 0 else 0
                        resid_abs = abs(resid)
                        
                        data_rows.append({
                            'genotype': genotype,
                            'obs': obs,
                            'pred': pred,
                            'resid': resid,
                            'resid_pct': resid_pct,
                            'resid_abs': resid_abs,
                            'index': i
                        })
                    
                    # Ordenar por ordem crescente (sequencial 1, 2, 3, ...)
                    data_rows.sort(key=lambda x: int(x['genotype']))
                    
                    # Adicionar linhas ordenadas
                    for row in data_rows:
                        values_data.append([
                            row['genotype'],
                            f'{row["obs"]:.4f}',
                            f'{row["pred"]:.4f}',
                            f'{row["resid"]:.4f}',
                            f'{row["resid_pct"]:.2f}%',
                            f'{row["resid_abs"]:.4f}'
                        ])
                    
                    df_values = pd.DataFrame(values_data[1:], columns=values_data[0])
                    df_values.to_excel(writer, sheet_name='Reg_6_Valores', index=False)
                    print("DEBUG: Planilha de valores observados vs estimados criada com sucesso!")
                except Exception as e:
                    print(f"DEBUG: Erro ao criar valores observados vs estimados: {e}")
                    import traceback
                    traceback.print_exc()
                
            print("DEBUG: Todas as seções de regressão processadas com sucesso!")
                
        except Exception as e:
            print(f"DEBUG: Erro geral na função de regressão: {e}")
            import traceback
            traceback.print_exc()
            logging.warning(f"Erro ao adicionar seções de regressão ao Excel: {e}")
    
    def _add_pca_sections_to_excel(self, writer):
        """Adiciona seções de PCA ao relatório Excel."""
        try:
            # Verificar se existem resultados de PCA
            print(f"DEBUG PCA: Verificando resultados de PCA...")
            if not hasattr(self, 'current_pca_results') or not self.current_pca_results:
                print("DEBUG PCA: Não há resultados de PCA disponíveis")
                return
                
            results = self.current_pca_results
            print(f"DEBUG PCA: Adicionando seções de PCA ao Excel com {results['n_components']} componentes...")
            
            # Limitar a 10 componentes conforme solicitado
            max_components = min(results['n_components'], 10)
            
            # 1. Autovalores e Variância Explicada
            if self.include_pca_eigenvalues.isChecked():
                eigenvalues_data = []
                eigenvalues_data.append(['Componente', 'Autovalor', 'Variância Explicada (%)', 'Variância Acumulada (%)'])
                
                cumulative_variance = 0
                for i in range(max_components):
                    eigenvalue = results['eigenvalues'][i]
                    variance_pct = results['explained_variance_ratio'][i] * 100
                    cumulative_variance += variance_pct
                    
                    eigenvalues_data.append([
                        f'PC{i+1}',
                        f'{eigenvalue:.6f}',
                        f'{variance_pct:.4f}%',
                        f'{cumulative_variance:.4f}%'
                    ])
                
                df_eigenvalues = pd.DataFrame(eigenvalues_data[1:], columns=eigenvalues_data[0])
                df_eigenvalues.to_excel(writer, sheet_name='PCA_1_Autovalores', index=False)
            
            # 2. Autovetores (Componentes Principais)
            if self.include_pca_eigenvectors.isChecked():
                eigenvectors_data = []
                header = ['Variável'] + [f'PC{i+1}' for i in range(max_components)]
                eigenvectors_data.append(header)
                
                for j, variable in enumerate(results['feature_names']):
                    row = [variable]
                    for i in range(max_components):
                        row.append(f'{results["eigenvectors"][i][j]:.6f}')
                    eigenvectors_data.append(row)
                
                df_eigenvectors = pd.DataFrame(eigenvectors_data[1:], columns=eigenvectors_data[0])
                df_eigenvectors.to_excel(writer, sheet_name='PCA_2_Autovetores', index=False)
            
            # 3. Cargas Fatoriais (Loadings)
            if self.include_pca_loadings.isChecked():
                loadings_data = []
                header = ['Variável'] + [f'PC{i+1}' for i in range(max_components)]
                loadings_data.append(header)
                
                for j, variable in enumerate(results['feature_names']):
                    row = [variable]
                    for i in range(max_components):
                        row.append(f'{results["loadings"][j][i]:.6f}')
                    loadings_data.append(row)
                
                df_loadings = pd.DataFrame(loadings_data[1:], columns=loadings_data[0])
                df_loadings.to_excel(writer, sheet_name='PCA_3_Loadings', index=False)
            
            # 4. Coordenadas dos Indivíduos
            if self.include_pca_coordinates.isChecked():
                coords_data = []
                header = ['Indivíduo'] + [f'PC{i+1}' for i in range(max_components)]
                coords_data.append(header)
                
                # Usar nomes dos genótipos diretamente do índice das coordenadas
                coordinates = results['coordinates']
                
                for i in range(len(coordinates)):
                    # Usar o índice das coordenadas que já contém os nomes reais dos genótipos
                    individual_name = str(coordinates.index[i])
                    
                    row = [individual_name]
                    for j in range(max_components):
                        row.append(f'{coordinates.iloc[i, j]:.6f}')
                    coords_data.append(row)
                
                df_coords = pd.DataFrame(coords_data[1:], columns=coords_data[0])
                df_coords.to_excel(writer, sheet_name='PCA_4_Coordenadas', index=False)
                
            print(f"DEBUG PCA: Seções de PCA adicionadas com sucesso!")
                
        except Exception as e:
            logging.warning(f"Erro ao adicionar seções de PCA ao Excel: {e}")
            print(f"DEBUG PCA: Erro - {e}")
    


    

    def create_input_tab(self):
        """Creates the data input tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Table for data input
        self.table_input = QTableWidget(15, 3) # Start with 15 rows, 3 columns
        self.table_input.setHorizontalHeaderLabels(self.original_column_names)
        self.table_input.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_input.itemChanged.connect(self.on_table_item_changed)
        
        # Enable context menu for copy/paste
        self.table_input.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_input.customContextMenuRequested.connect(self.show_input_table_context_menu)
        

        # Add row/column buttons
        btn_layout = QHBoxLayout()
        add_row_btn = QPushButton(GetFromLanguage("InputTab", 0))
        add_row_btn.clicked.connect(lambda: self.table_input.insertRow(self.table_input.rowCount()))
        remove_row_btn = QPushButton(GetFromLanguage("InputTab", 1))
        remove_row_btn.clicked.connect(self.remove_selected_row)
        btn_layout.addWidget(add_row_btn)
        btn_layout.addWidget(remove_row_btn)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)
        layout.addWidget(self.table_input)
        return tab

    def show_input_table_context_menu(self, pos):
        """Shows a context menu for the input table."""
        context_menu = QMenu(self)
        copy_action = context_menu.addAction(GetFromLanguage("InputTab", 2))
        paste_action = context_menu.addAction(GetFromLanguage("InputTab", 3))
        action = context_menu.exec_(self.table_input.mapToGlobal(pos))
        if action == copy_action:
            self.copy_input_table_selection()
        elif action == paste_action:
            self.paste_into_input_table()

    def copy_input_table_selection(self):
        """Copies the selected cells from the input table to the clipboard."""
        selection = self.table_input.selectedRanges()
        if not selection:
            return

        rows = []
        for r in range(selection[0].topRow(), selection[0].bottomRow() + 1):
            row_data = []
            for c in range(selection[0].leftColumn(), selection[0].rightColumn() + 1):
                item = self.table_input.item(r, c)
                row_data.append(item.text() if item else "")
            rows.append("\t".join(row_data))
        
        clipboard = QApplication.clipboard()
        clipboard.setText("\n".join(rows))

    def paste_into_input_table(self):
        """Pastes data from the clipboard into the input table."""
        selection = self.table_input.selectedRanges()
        if not selection:
            return

        clipboard = QApplication.clipboard()
        text = clipboard.text()
        rows = text.split('\n')

        start_row = selection[0].topRow()
        start_col = selection[0].leftColumn()

        for i, row_text in enumerate(rows):
            if not row_text:
                continue
            
            values = row_text.split('\t')
            for j, value in enumerate(values):
                row = start_row + i
                col = start_col + j
                if row < self.table_input.rowCount() and col < self.table_input.columnCount():
                    self.table_input.setItem(row, col, QTableWidgetItem(value))

    # --- Tab Creation Methods ---

    def create_input_tab(self):
        """Creates the 'Data Input' tab with a table and instructions."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        info_label = QLabel(GetFromLanguage("Dialogs", 43))
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("padding: 20px; background-color: #f0f0f0; border: 2px dashed #cccccc;")
        
        # Create table controls
        table_controls = QHBoxLayout()
        self.add_row_btn = QPushButton(GetFromLanguage("InputTab", 0))
        self.add_row_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowUp))
        self.add_row_btn.clicked.connect(self.add_table_row)
        
        self.remove_row_btn = QPushButton(GetFromLanguage("InputTab", 1))
        self.remove_row_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowDown))
        self.remove_row_btn.clicked.connect(self.remove_table_row)
        
        self.clear_cell_btn = QPushButton(GetFromLanguage("InputTab", 4))
        self.clear_cell_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogResetButton))
        self.clear_cell_btn.clicked.connect(self.clear_selected_cells)
        
        table_controls.addWidget(self.add_row_btn)
        table_controls.addWidget(self.remove_row_btn)
        table_controls.addWidget(self.clear_cell_btn)
        table_controls.addStretch()
        
        self.input_table = QTableWidget(20, 3)
        self.input_table.setHorizontalHeaderLabels(self.original_column_names)
        self.input_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Enable context menu and keyboard shortcuts
        self.input_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.input_table.customContextMenuRequested.connect(self.show_table_context_menu)
        
        # Add keyboard shortcuts
        delete_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self.input_table)
        delete_shortcut.activated.connect(self.clear_selected_cells)
        
        # Ctrl+C for copy
        copy_shortcut = QShortcut(QKeySequence.Copy, self.input_table)
        copy_shortcut.activated.connect(self.copy_selected_cells)
        
        # Ctrl+V for paste
        paste_shortcut = QShortcut(QKeySequence.Paste, self.input_table)
        paste_shortcut.activated.connect(self.paste_cells)
        
        # Ctrl+Plus for add row
        add_row_shortcut = QShortcut(QKeySequence("Ctrl++"), self.input_table)
        add_row_shortcut.activated.connect(self.add_table_row)
        
        # Ctrl+Minus for remove row
        remove_row_shortcut = QShortcut(QKeySequence("Ctrl+-"), self.input_table)
        remove_row_shortcut.activated.connect(self.remove_table_row)
        
        # Enable cell editing enhancements
        self.input_table.setSelectionBehavior(QTableWidget.SelectItems)
        self.input_table.setSelectionMode(QTableWidget.ExtendedSelection)
        
        # Conectar sinal para atualizar UI quando dados mudam
        self.input_table.itemChanged.connect(self.on_table_data_changed)
        
        layout.addWidget(info_label)
        layout.addLayout(table_controls)
        layout.addWidget(self.input_table)
        return tab

    def create_indices_tab(self):
        """Creates the 'Calculated Indices' tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.indices_table = QTableWidget()
        self.indices_table.setSortingEnabled(True)
        self.setup_table_copy_functionality(self.indices_table, "indices")
        layout.addWidget(self.indices_table)
        return tab

    def create_rankings_tab(self):
        """Creates the 'Rankings' tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.rank_table = QTableWidget()
        self.rank_table.setSortingEnabled(True)
        self.setup_table_copy_functionality(self.rank_table, "rankings")
        layout.addWidget(self.rank_table)
        return tab

    def create_statistics_tab(self):
        """Creates the 'Statistical Analysis' tab with numerical tables and analyses."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create a splitter for better organization
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)
        
        # Controls section
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        
        # Analysis options
        analysis_group = QGroupBox(GetFromLanguage("StatisticsTab", 0))
        analysis_group.setStyleSheet(self.get_professional_group_style())
        analysis_layout = QFormLayout(analysis_group)
        
        self.stats_use_rankings = QCheckBox(GetFromLanguage("StatisticsTab", 1))
        self.stats_use_rankings.setToolTip("Usa rankings ao invés de índices para análises")
        analysis_layout.addRow(GetFromLanguage("StatisticsTab", 2), self.stats_use_rankings)
        
        self.correlation_method_stats = QComboBox()
        self.correlation_method_stats.addItems(['pearson', 'spearman', 'kendall'])
        analysis_layout.addRow(GetFromLanguage("StatisticsTab", 3), self.correlation_method_stats)
        
        self.refresh_stats_btn = QPushButton(GetFromLanguage("StatisticsTab", 4))
        self.refresh_stats_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.refresh_stats_btn.clicked.connect(self.refresh_statistical_analyses)
        analysis_layout.addRow(self.refresh_stats_btn)
        
        controls_layout.addWidget(analysis_group)
        
        # Clustering options (integrated)
        cluster_group = QGroupBox("Clustering")
        cluster_group.setStyleSheet(self.get_professional_group_style())
        cluster_layout = QFormLayout(cluster_group)
        
        self.cluster_method_combo = QComboBox()
        self.cluster_method_combo.addItems(['K-Means', 'Agglomerativo'])
        cluster_layout.addRow(GetFromLanguage("StatisticsTab", 5), self.cluster_method_combo)
        
        self.cluster_number_combo = QComboBox()
        self.cluster_number_combo.addItem(GetFromLanguage("StatisticsTab", 6))  # Placeholder for "Select number of clusters"
        for i in range(2, 11):
            self.cluster_number_combo.addItem(str(i))
        cluster_layout.addRow(GetFromLanguage("StatisticsTab", 7), self.cluster_number_combo)
        
        self.cluster_button = QPushButton(GetFromLanguage("StatisticsTab", 8))
        self.cluster_button.clicked.connect(self.start_clustering_with_options)
        cluster_layout.addRow(self.cluster_button)
        
               
        controls_layout.addWidget(cluster_group)
        controls_layout.addStretch()
        splitter.addWidget(controls_widget)
        
        # Results area with tabs
        results_widget = QTabWidget()
        
        # Tab 1: Descriptive Statistics
        self.descriptive_table = QTableWidget()
        self.setup_table_copy_functionality(self.descriptive_table, "descriptive")
        results_widget.addTab(self.descriptive_table, GetFromLanguage("StatisticsTab", 9))
        
        # Tab 2: Correlation Matrix
        self.correlation_table = QTableWidget()
        self.setup_table_copy_functionality(self.correlation_table, "correlation")
        results_widget.addTab(self.correlation_table, GetFromLanguage("StatisticsTab", 10))
        
        # Tab 3: Normality Tests
        self.normality_table = QTableWidget()
        self.setup_table_copy_functionality(self.normality_table, "normality")
        results_widget.addTab(self.normality_table, GetFromLanguage("StatisticsTab", 11))
        
        # Tab 4: Outliers Detection
        self.outliers_table = QTableWidget()
        self.setup_table_copy_functionality(self.outliers_table, "outliers")
        results_widget.addTab(self.outliers_table, GetFromLanguage("StatisticsTab", 12))
        
        # Tab 5: Cluster Analysis
        self.cluster_analysis_table = QTableWidget()
        self.setup_table_copy_functionality(self.cluster_analysis_table, "cluster")
        results_widget.addTab(self.cluster_analysis_table, GetFromLanguage("StatisticsTab", 13))
        
        
        # Tab 7: Heritability Analysis (NOVO - do artigo)
        self.heritability_table = QTableWidget()
        self.setup_table_copy_functionality(self.heritability_table, "heritability")
        results_widget.addTab(self.heritability_table, GetFromLanguage("StatisticsTab", 14))
        
        
        # Tab 9: Stability Analysis (NOVO - do artigo)
        # Aba de Estabilidade removida conforme solicitado

        # Tab 10: MFV Analysis
        self.mfv_table = QTableWidget()
        self.setup_table_copy_functionality(self.mfv_table, "mfv")
        results_widget.addTab(self.mfv_table, GetFromLanguage("StatisticsTab", 20))
        
        splitter.addWidget(results_widget)
        splitter.setSizes([150, 600])
        
        return tab

    def init_graph_controls(self):
        """Initializes all widgets for the unified graph control panel."""
        # --- Group Boxes ---
        self.graph_type_group = QGroupBox(GetFromLanguage("GraphTabs", 0))
        self.vars_group = QGroupBox(GetFromLanguage("GraphTabs", 1))
        self.pca_group = QGroupBox(GetFromLanguage("GraphTabs", 2))
        self.dendro_group = QGroupBox(GetFromLanguage("GraphTabs", 3))
        self.heatmap_group = QGroupBox(GetFromLanguage("GraphTabs", 4))
        self.style_group = QGroupBox(GetFromLanguage("GraphTabs", 5))
        self.export_group = QGroupBox(GetFromLanguage("GraphTabs", 6))

        # --- Type Selection ---
        self.graph_type_combo = QComboBox()
        self.graph_type_combo.addItems([
            GetFromLanguage("GraphTypes", 0), GetFromLanguage("GraphTypes", 1), GetFromLanguage("GraphTypes", 2), GetFromLanguage("GraphTypes", 3),
            GetFromLanguage("GraphTypes", 4), GetFromLanguage("GraphTypes", 5), GetFromLanguage("GraphTypes", 6), GetFromLanguage("GraphTypes", 7)
        ])
        self.use_rankings_graph = QCheckBox(GetFromLanguage("GraphTypes", 8))

        # --- Variable Selection (Single) ---
        self.single_var_widget = QWidget()
        single_var_layout = QFormLayout(self.single_var_widget)
        single_var_layout.setContentsMargins(0, 0, 0, 0)
        self.single_var_combo = QComboBox()
        single_var_layout.addRow("Variável:", self.single_var_combo)

        # --- Variable Selection (Multi) ---
        self.multi_var_widget = QWidget()
        multi_var_layout = QHBoxLayout(self.multi_var_widget)
        multi_var_layout.setContentsMargins(0, 0, 0, 0)
        
        # Available list
        available_layout = QVBoxLayout()
        available_layout.addWidget(QLabel("Disponíveis:"))
        self.multi_var_list_available = QListWidget()
        self.multi_var_list_available.setSelectionMode(QListWidget.ExtendedSelection)
        available_layout.addWidget(self.multi_var_list_available)
        
        # Buttons
        buttons_layout = QVBoxLayout()
        buttons_layout.addStretch()
        self.add_var_btn = QPushButton(">")
        self.add_all_vars_btn = QPushButton(">>")
        self.remove_var_btn = QPushButton("<")
        self.remove_all_vars_btn = QPushButton("<<")
        buttons_layout.addWidget(self.add_var_btn)
        buttons_layout.addWidget(self.add_all_vars_btn)
        buttons_layout.addWidget(self.remove_var_btn)
        buttons_layout.addWidget(self.remove_all_vars_btn)
        buttons_layout.addStretch()

        # Selected list
        selected_layout = QVBoxLayout()
        selected_layout.addWidget(QLabel("Selecionadas:"))
        self.multi_var_list_selected = QListWidget()
        self.multi_var_list_selected.setSelectionMode(QListWidget.ExtendedSelection)
        selected_layout.addWidget(self.multi_var_list_selected)

        multi_var_layout.addLayout(available_layout)
        multi_var_layout.addLayout(buttons_layout)
        multi_var_layout.addLayout(selected_layout)

        # --- Specific Parameters ---
        # Detach from old layout if they were initialized before
        if hasattr(self, 'pca_components_spin'):
            self.pca_components_spin.setParent(None)
            self.pca_scaling_check.setParent(None)
            self.dendro_method_combo.setParent(None)
            self.dendro_metric_combo.setParent(None)
            self.dendro_groups_spin.setParent(None)
            self.dendro_color_groups_check.setParent(None)
            self.heatmap_method_combo.setParent(None)
            self.heatmap_values_check.setParent(None)
        
        # --- Scatter Plot Axis Selection ---
        self.scatter_group = QGroupBox(GetFromLanguage("GraphTabs", 15))
        self.scatter_x_combo = QComboBox()
        self.scatter_y_combo = QComboBox()
        scatter_layout = QFormLayout(self.scatter_group)
        scatter_layout.addRow(GetFromLanguage("GraphTabs", 16), self.scatter_x_combo)
        scatter_layout.addRow(GetFromLanguage("GraphTabs", 17), self.scatter_y_combo)

        # --- 3D Graph Controls ---
        self.graph_3d_group = QGroupBox(GetFromLanguage("GraphTabs", 18))
        self.graph_3d_x_combo = QComboBox()
        self.graph_3d_y_combo = QComboBox()
        self.graph_3d_z_combo = QComboBox()
        graph_3d_layout = QFormLayout(self.graph_3d_group)
        graph_3d_layout.addRow("Eixo X:", self.graph_3d_x_combo)
        graph_3d_layout.addRow("Eixo Y:", self.graph_3d_y_combo)
        graph_3d_layout.addRow("Eixo Z:", self.graph_3d_z_combo)

        # --- Style and Titles ---
        self.title_input = QLineEdit()
        self.xlabel_input = QLineEdit()
        self.ylabel_input = QLineEdit()
        self.zlabel_input = QLineEdit()  # Para gráficos 3D
        self.hue_combo = QComboBox()
        self.hue_combo.addItems(['Nenhum', 'Classificação por RP', 'Classificação por MFV'])
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(['viridis', 'Set1', 'plasma', 'tab10', 'coolwarm', 'default'])
        self.show_labels_check = QCheckBox(GetFromLanguage("GraphTabs", 13))

        # --- Export ---
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(['PNG', 'PDF', 'SVG', 'JPG', 'EPS', 'TIFF'])
        self.export_format_combo.setToolTip('Formato de arquivo para exportação')
        self.export_dpi_spin = QSpinBox()
        self.export_dpi_spin.setRange(72, 600)
        self.export_dpi_spin.setValue(300)
        self.export_dpi_spin.setSuffix(" DPI")
        self.export_dpi_spin.setToolTip('Resolução da imagem (pontos por polegada)')

        # --- Actions ---
        self.auto_update_check = QCheckBox(GetFromLanguage("GraphTabs", 18))
        self.auto_update_check.setChecked(True)  # Enable by default
        self.generate_graph_button = QPushButton(GetFromLanguage("GraphTabs", 14))
        self.export_graph_button = QPushButton(GetFromLanguage("GraphTabs", 15))
        
    def create_graphics_tab(self):
        """Creates the new unified and contextual graphics tab."""
        tab = QWidget()
        main_layout = QHBoxLayout(tab)

        # --- Left Control Panel ---
        control_panel_scroll = QScrollArea()
        control_panel_scroll.setWidgetResizable(True)
        control_panel_scroll.setMinimumWidth(350)
        control_panel_scroll.setMaximumWidth(450)
        
        control_panel = QWidget()
        control_panel_scroll.setWidget(control_panel)
        
        panel_layout = QVBoxLayout(control_panel)

        # GroupBox 1: Type
        type_layout = QFormLayout(self.graph_type_group)
        type_layout.addRow("Tipo:", self.graph_type_combo)
        # Criar widget container para o checkbox alinhado à esquerda
        ranking_widget = QWidget()
        ranking_layout = QHBoxLayout(ranking_widget)
        ranking_layout.setContentsMargins(0, 0, 0, 0)
        ranking_layout.addWidget(self.use_rankings_graph)
        ranking_layout.addStretch()
        type_layout.addRow(ranking_widget)
        panel_layout.addWidget(self.graph_type_group)

        # GroupBox 2: Variables (contains switchable widgets)
        vars_layout = QVBoxLayout(self.vars_group)
        vars_layout.addWidget(self.single_var_widget)
        vars_layout.addWidget(self.multi_var_widget)
        panel_layout.addWidget(self.vars_group)

        # GroupBox 3: Specific Params
        pca_layout = QFormLayout(self.pca_group)
        pca_layout.addRow("N° Componentes:", self.pca_components_spin)
        pca_layout.addRow("Eixo X:", self.pca_component_x_combo)
        pca_layout.addRow("Eixo Y:", self.pca_component_y_combo)
        pca_layout.addRow("Matriz:", self.pca_matrix_combo)
        
        # Criar widget container para o checkbox alinhado à esquerda
        checkbox_widget = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_widget)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        checkbox_layout.addWidget(self.pca_show_variance_check)
        checkbox_layout.addStretch()
        
        # Adicionar o checkbox ocupando toda a largura, sem label
        pca_layout.addRow(checkbox_widget)
        panel_layout.addWidget(self.pca_group)

        dendro_layout = QFormLayout(self.dendro_group)
        dendro_layout.addRow("Método:", self.dendro_method_combo)
        dendro_layout.addRow("Métrica:", self.dendro_metric_combo)
        dendro_layout.addRow("Nº Grupos:", self.dendro_groups_spin)
        
        # Criar widget container para o checkbox alinhado à esquerda
        dendro_color_widget = QWidget()
        dendro_color_layout = QHBoxLayout(dendro_color_widget)
        dendro_color_layout.setContentsMargins(0, 0, 0, 0)
        dendro_color_layout.addWidget(self.dendro_color_groups_check)
        dendro_color_layout.addStretch()
        dendro_layout.addRow(dendro_color_widget)
        panel_layout.addWidget(self.dendro_group)

        heatmap_layout = QFormLayout(self.heatmap_group)
        heatmap_layout.addRow("Método:", self.heatmap_method_combo)
        # Criar widget container para o checkbox alinhado à esquerda
        heatmap_check_widget = QWidget()
        heatmap_check_layout = QHBoxLayout(heatmap_check_widget)
        heatmap_check_layout.setContentsMargins(0, 0, 0, 0)
        heatmap_check_layout.addWidget(self.heatmap_values_check)
        heatmap_check_layout.addStretch()
        heatmap_layout.addRow(heatmap_check_widget)
        panel_layout.addWidget(self.heatmap_group)
        panel_layout.addWidget(self.scatter_group)
        panel_layout.addWidget(self.graph_3d_group)

        # GroupBox 4: Style
        style_layout = QFormLayout(self.style_group)
        style_layout.addRow(GetFromLanguage("GraphTabs", 7), self.title_input)
        style_layout.addRow(GetFromLanguage("GraphTabs", 8), self.xlabel_input)
        style_layout.addRow(GetFromLanguage("GraphTabs", 9), self.ylabel_input)
        style_layout.addRow(GetFromLanguage("GraphTabs", 10), self.zlabel_input)
        style_layout.addRow(GetFromLanguage("GraphTabs", 11), self.hue_combo)
        style_layout.addRow(GetFromLanguage("GraphTabs", 12), self.palette_combo)
        # Criar widget container para o checkbox alinhado à esquerda
        labels_check_widget = QWidget()
        labels_check_layout = QHBoxLayout(labels_check_widget)
        labels_check_layout.setContentsMargins(0, 0, 0, 0)
        labels_check_layout.addWidget(self.show_labels_check)
        labels_check_layout.addStretch()
        style_layout.addRow(labels_check_widget)
        panel_layout.addWidget(self.style_group)
        
        # GroupBox 5: Export
        export_layout = QFormLayout(self.export_group)
        export_layout.addRow(GetFromLanguage("GraphTabs", 16), self.export_format_combo)
        export_layout.addRow(GetFromLanguage("GraphTabs", 17), self.export_dpi_spin)
        panel_layout.addWidget(self.export_group)

        panel_layout.addStretch()

        # --- Bottom Action Buttons ---
        action_layout = QHBoxLayout()
        action_layout.addWidget(self.generate_graph_button)
        action_layout.addWidget(self.export_graph_button)
        panel_layout.addLayout(action_layout)
        # Criar widget container para o checkbox alinhado à esquerda
        auto_update_widget = QWidget()
        auto_update_layout = QHBoxLayout(auto_update_widget)
        auto_update_layout.setContentsMargins(0, 0, 0, 0)
        auto_update_layout.addWidget(self.auto_update_check)
        auto_update_layout.addStretch()
        panel_layout.addWidget(auto_update_widget)

        # --- Right Graph Preview Area ---
        preview_widget = QWidget()
        preview_widget.setStyleSheet("background: white; border: 1px solid #d0d0d0; border-radius: 6px;")
        self.graph_preview_layout = QVBoxLayout(preview_widget)
        self.graph_preview_layout.setContentsMargins(10, 10, 10, 10)
        
        placeholder = QLabel("Visualização de Gráficos\n\nSelecione um tipo de gráfico e clique em 'Gerar Gráfico'")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: #666; font-size: 14px; padding: 60px 20px; font-weight: 500;")
        self.graph_preview_layout.addWidget(placeholder)

        # --- Splitter ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(control_panel_scroll)
        splitter.addWidget(preview_widget)
        splitter.setSizes([350, 700])
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        # --- Connect Signals ---
        self.graph_type_combo.currentTextChanged.connect(self.update_graph_controls)
        self.graph_type_combo.currentTextChanged.connect(self._handle_auto_update)
        self.generate_graph_button.clicked.connect(self.generate_graph)
        self.export_graph_button.clicked.connect(self.export_graph)
        self.use_rankings_graph.stateChanged.connect(self.update_variable_lists)
        
        # Connect auto-update checkbox
        self.auto_update_check.stateChanged.connect(
            lambda state: self.toggle_auto_update(state == 2)  # Qt.Checked = 2
        )
        
        # Initialize auto-update state
        self.auto_update_enabled = self.auto_update_check.isChecked()
        
        # Auto-update connections - only include controls that exist
        controls_to_update = [
            self.use_rankings_graph, self.single_var_combo, self.pca_components_spin,
            self.dendro_method_combo, self.dendro_metric_combo,
            self.dendro_groups_spin, self.dendro_color_groups_check,
            self.heatmap_method_combo, self.heatmap_values_check, self.hue_combo,
            self.palette_combo, self.show_labels_check, self.title_input,
            self.xlabel_input, self.ylabel_input, self.zlabel_input,
            # Add scatter plot controls
            self.scatter_x_combo, self.scatter_y_combo,
            # Add 3D plot controls
            self.graph_3d_x_combo, self.graph_3d_y_combo, self.graph_3d_z_combo,
            # Add new PCA controls
            self.pca_component_x_combo, self.pca_component_y_combo,
            self.pca_matrix_combo, self.pca_show_variance_check
        ]
        
        # Add font controls if they exist
        font_controls = ['title_font_combo', 'axes_font_combo', 'tick_font_combo',
                        'legend_font_combo', 'annotation_font_combo', 'general_font_combo']
        for font_control_name in font_controls:
            if hasattr(self, font_control_name):
                controls_to_update.append(getattr(self, font_control_name))
        for control in controls_to_update:
            if isinstance(control, QComboBox):
                control.currentTextChanged.connect(self._handle_auto_update)
            elif isinstance(control, QLineEdit):
                control.textChanged.connect(self._handle_auto_update)
            elif isinstance(control, QCheckBox):
                control.stateChanged.connect(self._handle_auto_update)
            elif isinstance(control, QSpinBox):
                control.valueChanged.connect(self._handle_auto_update)


        # Variable list buttons
        self.add_var_btn.clicked.connect(self._add_var_to_selection)
        self.add_all_vars_btn.clicked.connect(lambda: self._move_all_vars(self.multi_var_list_available, self.multi_var_list_selected))
        self.remove_var_btn.clicked.connect(self._remove_var_from_selection)
        self.remove_all_vars_btn.clicked.connect(lambda: self._move_all_vars(self.multi_var_list_selected, self.multi_var_list_available))

        self.update_graph_controls() # Set initial state
        return tab

    def _handle_auto_update(self):
        """Checks if auto-update is on and generates the graph."""
        if self.auto_update_check.isChecked() and hasattr(self, 'indices') and self.indices is not None and not self.indices.empty:
            self.generate_graph()

    def _add_var_to_selection(self):
        """Moves selected items from available to selected list."""
        for item in self.multi_var_list_available.selectedItems():
            self.multi_var_list_selected.addItem(self.multi_var_list_available.takeItem(self.multi_var_list_available.row(item)))
        self._handle_auto_update()

    def _remove_var_from_selection(self):
        """Moves selected items from selected to available list."""
        for item in self.multi_var_list_selected.selectedItems():
            self.multi_var_list_available.addItem(self.multi_var_list_selected.takeItem(self.multi_var_list_selected.row(item)))
        self._handle_auto_update()
        
    def _move_all_vars(self, source_list, dest_list):
        """Moves all items from source list to destination list."""
        while source_list.count() > 0:
            dest_list.addItem(source_list.takeItem(0))
        self._handle_auto_update()

    def update_graph_controls(self):
        """Shows or hides graph controls based on the selected graph type."""
        graph_type = self.graph_type_combo.currentText()

        # Hide all specific groups first
        self.vars_group.setVisible(False)
        self.pca_group.setVisible(False)
        self.dendro_group.setVisible(False)
        self.heatmap_group.setVisible(False)
        self.scatter_group.setVisible(False) # Hide scatter group by default
        self.graph_3d_group.setVisible(False) # Hide 3D group by default
        self.xlabel_input.setEnabled(True)
        self.ylabel_input.setEnabled(True)
        self.zlabel_input.setEnabled(False) # Z-label disabled by default

        if graph_type == 'Gráfico de Dispersão':
            self.scatter_group.setVisible(True)
        elif graph_type == 'Gráfico 3D':
            self.graph_3d_group.setVisible(True)
            self.zlabel_input.setEnabled(True)
        elif graph_type == 'PCA Biplot':
            self.pca_group.setVisible(True)
        elif graph_type == 'Dendrograma':
            self.dendro_group.setVisible(True)
            self.ylabel_input.setEnabled(False) # Dendrogram controls its own labels
        elif graph_type == 'Gráfico de Barras':
            self.vars_group.setVisible(True)
            self.single_var_widget.setVisible(False)
            self.multi_var_widget.setVisible(True)
        elif graph_type == 'Histograma':
            self.vars_group.setVisible(True)
            self.single_var_widget.setVisible(False)
            self.multi_var_widget.setVisible(True)
            self.ylabel_input.setEnabled(False) # Histogram controls its own y-axis
        elif graph_type == 'Box Plot':
            self.vars_group.setVisible(True)
            self.single_var_widget.setVisible(False)
            self.multi_var_widget.setVisible(True)
        elif graph_type == 'Heatmap de Correlação':
            self.vars_group.setVisible(True)
            self.single_var_widget.setVisible(False)
            self.multi_var_widget.setVisible(True)
            self.heatmap_group.setVisible(True)
            self.xlabel_input.setEnabled(False)
            self.ylabel_input.setEnabled(False)
        
        self._handle_auto_update()

    def get_modern_combo_style(self):
        """Returns modern combobox styling."""
        return """
            QComboBox {
                padding: 8px 12px; border: 2px solid #e0e0e0; border-radius: 6px;
                background: white; font-size: 13px; min-height: 20px;
            }
            QComboBox:focus { border-color: #4CAF50; }
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow { image: url(none); }
        """
    
    def get_professional_group_style(self):
        """Returns professional group box styling."""
        return """
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                color: #333333;
                border: 1px solid #cccccc;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
                background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 6px 0 6px;
                background: white;
            }
        """
    
    def get_professional_combo_style(self):
        """Returns professional combobox styling."""
        return """
            QComboBox {
                padding: 6px 10px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background: white;
                font-size: 11px;
                min-height: 18px;
            }
            QComboBox:focus {
                border-color: #2E7D32;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                width: 12px;
                height: 12px;
            }
        """
    
    def get_professional_spinbox_style(self):
        """Returns professional spinbox styling."""
        return """
            QSpinBox {
                padding: 6px 8px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background: white;
                font-size: 11px;
                min-height: 18px;
            }
            QSpinBox:focus {
                border-color: #2E7D32;
            }
        """
    
    def get_professional_lineedit_style(self):
        """Returns professional line edit styling."""
        return """
            QLineEdit {
                padding: 6px 8px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background: white;
                font-size: 11px;
                min-height: 18px;
            }
            QLineEdit:focus {
                border-color: #2E7D32;
            }
        """

    def create_regression_tab(self):
        """Cria a aba de Análise de Regressão Múltipla."""
        tab = QWidget()
        main_layout = QHBoxLayout(tab)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # --- Left Control Panel (similar to graphics tab) ---
        control_panel = QWidget()
        control_panel.setMaximumWidth(450)
        control_panel.setStyleSheet("background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px;")
        
        control_panel_scroll = QScrollArea()
        control_panel_scroll.setWidget(control_panel)
        control_panel_scroll.setWidgetResizable(True)
        control_panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        control_panel_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        control_panel_scroll.setMinimumWidth(450)
        control_panel_scroll.setMaximumWidth(450)
        
        panel_layout = QVBoxLayout(control_panel)
        panel_layout.setContentsMargins(15, 15, 15, 15)
        panel_layout.setSpacing(15)
        
        # Título do painel
        title_label = QLabel(GetFromLanguage("MultipleRegressionTab", 0))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #333; margin: 0; padding: 10px;")
        panel_layout.addWidget(title_label)
        
        # Informações
        info_label = QLabel(GetFromLanguage("MultipleRegressionTab", 10))
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("padding: 10px; color: #666; font-size: 12px; background: #e8f5e8; border-radius: 4px;")
        panel_layout.addWidget(info_label)
        
        # Grupo de Configurações da regressão
        config_group = QGroupBox(GetFromLanguage("MultipleRegressionTab", 11))
        config_group.setStyleSheet(self.get_professional_group_style())
        config_layout = QFormLayout(config_group)
        config_layout.setSpacing(12)
        config_layout.setContentsMargins(15, 20, 15, 15)
        
        # Variável dependente (Y)
        self.regression_y_var = QComboBox()
        self.regression_y_var.setStyleSheet(self.get_professional_combo_style())
        config_layout.addRow(GetFromLanguage("MultipleRegressionTab", 1), self.regression_y_var)
        
        # Variáveis independentes (X) - Lista com checkboxes
        self.regression_x_vars = QListWidget()
        self.regression_x_vars.setSelectionMode(QListWidget.MultiSelection)
        self.regression_x_vars.setMaximumHeight(120)
        self.regression_x_vars.setStyleSheet("""
            QListWidget {
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background: white;
                font-size: 12px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 4px 8px;
                border-radius: 3px;
            }
            QListWidget::item:selected {
                background: #4CAF50;
                color: white;
            }
        """)
        config_layout.addRow(GetFromLanguage("MultipleRegressionTab", 2), self.regression_x_vars)
        panel_layout.addWidget(config_group)
        
        # Grupo de Opções
        options_group = QGroupBox(GetFromLanguage("MultipleRegressionTab", 12))
        options_group.setStyleSheet(self.get_professional_group_style())
        options_layout = QVBoxLayout(options_group)
        options_layout.setSpacing(8)
        options_layout.setContentsMargins(15, 20, 15, 15)
        
        self.include_intercept = QCheckBox(GetFromLanguage("MultipleRegressionTab", 13))
        self.include_intercept.setChecked(True)
        self.include_intercept.setStyleSheet("QCheckBox { font-size: 12px; padding: 4px; }")
        options_layout.addWidget(self.include_intercept)
        
        self.standardize_vars = QCheckBox(GetFromLanguage("MultipleRegressionTab", 14))
        self.standardize_vars.setChecked(False)
        self.standardize_vars.setStyleSheet("QCheckBox { font-size: 12px; padding: 4px; }")
        options_layout.addWidget(self.standardize_vars)
        
        self.show_residuals = QCheckBox(GetFromLanguage("MultipleRegressionTab", 15))
        self.show_residuals.setChecked(True)
        self.show_residuals.setStyleSheet("QCheckBox { font-size: 12px; padding: 4px; }")
        options_layout.addWidget(self.show_residuals)
        
        panel_layout.addWidget(options_group)
        
        panel_layout.addStretch()
        
        # Botões de ação
        action_layout = QHBoxLayout()
        
        self.update_regression_vars_btn = QPushButton(GetFromLanguage("MultipleRegressionTab", 16))
        self.update_regression_vars_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.update_regression_vars_btn.clicked.connect(self.update_regression_variables)
        self.update_regression_vars_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 12px;
                border: 2px solid #2196F3;
                border-radius: 6px;
                background: white;
                color: #2196F3;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #2196F3;
                color: white;
            }
        """)
        action_layout.addWidget(self.update_regression_vars_btn)
        
        self.run_regression_btn = QPushButton(GetFromLanguage("MultipleRegressionTab", 3))
        self.run_regression_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.run_regression_btn.clicked.connect(self.run_regression_analysis)
        self.run_regression_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 15px;
                border: none;
                border-radius: 6px;
                background: #4CAF50;
                color: white;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #45a049;
            }
        """)
        action_layout.addWidget(self.run_regression_btn)
        
        panel_layout.addLayout(action_layout)
        
        # --- Right Results Area (main display area) ---
        results_widget = QWidget()
        results_widget.setStyleSheet("background: white; border: 1px solid #d0d0d0; border-radius: 6px;")
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(15, 15, 15, 15)
        
        # Título da área de resultados
        results_title = QLabel(GetFromLanguage("MultipleRegressionTab", 4))
        results_title.setAlignment(Qt.AlignCenter)
        results_title.setStyleSheet("color: #333; margin-bottom: 15px; padding: 10px;")
        results_layout.addWidget(results_title)
        
        # TabWidget para sub-abas de resultados
        self.regression_results_tabs = QTabWidget()
        self.regression_results_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                background: white;
            }
            QTabBar::tab {
                padding: 10px 20px;
                margin: 2px;
                border-radius: 4px;
                background: #f5f5f5;
                color: #666;
            }
            QTabBar::tab:selected {
                background: #4CAF50;
                color: white;
                font-weight: bold;
            }
        """)
        
        # Aba 1: Resumo do Modelo
        self.regression_summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.regression_summary_tab)
        summary_layout.setContentsMargins(10, 10, 10, 10)
        self.regression_summary_text = QTextEdit()
        self.regression_summary_text.setReadOnly(True)
        self.regression_summary_text.setFont(QFont("Courier", 9))
        self.regression_summary_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background: #fafafa;
                padding: 10px;
            }
        """)
        summary_layout.addWidget(self.regression_summary_text)
        self.regression_results_tabs.addTab(self.regression_summary_tab, GetFromLanguage("MultipleRegressionTab", 17))
        self.regression_results_tabs.setStyleSheet("""""")
        
        # Aba 2: Coeficientes
        self.regression_coef_tab = QWidget()
        coef_layout = QVBoxLayout(self.regression_coef_tab)
        coef_layout.setContentsMargins(10, 10, 10, 10)
        self.regression_coef_table = QTableWidget()
        self.setup_table_copy_functionality(self.regression_coef_table, "regression_coef")
        self.regression_coef_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background: white;
                gridline-color: #f0f0f0;
            }
            QHeaderView::section {
                background: #f8f9fa;
                border: 1px solid #e0e0e0;
                padding: 8px;
                font-weight: bold;
            }
        """)
        coef_layout.addWidget(self.regression_coef_table)
        self.regression_results_tabs.addTab(self.regression_coef_tab, GetFromLanguage("MultipleRegressionTab", 18))
        
        # Aba 3: Diagnósticos
        self.regression_diag_tab = QWidget()
        diag_layout = QVBoxLayout(self.regression_diag_tab)
        diag_layout.setContentsMargins(10, 10, 10, 10)
        self.regression_diag_table = QTableWidget()
        self.setup_table_copy_functionality(self.regression_diag_table, "regression_diag")
        self.regression_diag_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background: white;
                gridline-color: #f0f0f0;
            }
            QHeaderView::section {
                background: #f8f9fa;
                border: 1px solid #e0e0e0;
                padding: 8px;
                font-weight: bold;
            }
        """)
        diag_layout.addWidget(self.regression_diag_table)
        self.regression_results_tabs.addTab(self.regression_diag_tab, GetFromLanguage("MultipleRegressionTab", 19))
        
        # Aba 4: Análise de Resíduos
        self.regression_residuals_tab = QWidget()
        residuals_layout = QVBoxLayout(self.regression_residuals_tab)
        residuals_layout.setContentsMargins(10, 10, 10, 10)
        
        # Sub-tabWidget para análise de resíduos
        self.residuals_subtab_widget = QTabWidget()
        self.residuals_subtab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                background: white;
            }
            QTabBar::tab {
                background: #f8f8f8;
                padding: 8px 12px;
                margin-right: 2px;
                border: 1px solid #d0d0d0;
                border-bottom: none;
                border-radius: 6px 6px 0 0;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom: 1px solid white;
            }
        """)
        
        # Sub-aba 1: Estatísticas dos Resíduos
        self.residuals_stats_tab = QWidget()
        stats_layout = QVBoxLayout(self.residuals_stats_tab)
        stats_layout.setContentsMargins(10, 10, 10, 10)
        self.regression_residuals_text = QTextEdit()
        self.regression_residuals_text.setReadOnly(True)
        self.regression_residuals_text.setFont(QFont("Courier", 9))
        self.regression_residuals_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background: #fafafa;
                padding: 10px;
            }
        """)
        stats_layout.addWidget(self.regression_residuals_text)
        self.residuals_subtab_widget.addTab(self.residuals_stats_tab, GetFromLanguage("MultipleRegressionTab", 20))
        
        # Sub-aba 2: Tabela de Valores Detalhados
        self.residuals_table_tab = QWidget()
        table_layout = QVBoxLayout(self.residuals_table_tab)
        table_layout.setContentsMargins(10, 10, 10, 10)
        
        # Tabela para valores observados, estimados e resíduos
        self.residuals_values_table = QTableWidget()
        self.setup_table_copy_functionality(self.residuals_values_table, "residuals_values")
        self.residuals_values_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                gridline-color: #f0f0f0;
                background: white;
                selection-background-color: #b6d7ff;
            }
            QHeaderView::section {
                background: #f8f9fa;
                border: 1px solid #e0e0e0;
                padding: 8px;
                font-weight: bold;
            }
        """)
        self.residuals_values_table.setSortingEnabled(True)
        self.residuals_values_table.setAlternatingRowColors(True)
        table_layout.addWidget(self.residuals_values_table)
        
        self.residuals_subtab_widget.addTab(self.residuals_table_tab, GetFromLanguage("MultipleRegressionTab", 21))
        
        residuals_layout.addWidget(self.residuals_subtab_widget)
        self.regression_results_tabs.addTab(self.regression_residuals_tab, GetFromLanguage("MultipleRegressionTab", 22))
        
        results_layout.addWidget(self.regression_results_tabs, 1)
        
        # Status
        self.regression_status_label = QLabel("Pronto para análise. Selecione as variáveis e clique em 'Executar Regressão'.")
        self.regression_status_label.setStyleSheet("padding: 10px; color: #666; font-size: 12px; background: #f0f8ff; border-radius: 4px; margin-top: 10px;")
        self.regression_status_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.regression_status_label)
        
        # --- Splitter ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(control_panel_scroll)
        splitter.addWidget(results_widget)
        splitter.setSizes([350, 700])
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)
        
        return tab

    # === IMPLEMENTAÇÃO COMPLETA DA PCA PROFISSIONAL ===

    # A aba PCA foi totalmente redesenhada com:
    # 1. Interface profissional sem emojis
    # 2. Visualização unificada de até 10 componentes
    # 3. Três tabelas principais: Autovalores, Loadings e Coordenadas
    # 4. Cada tabela mostra TODOS os componentes selecionados de uma vez
    # 5. Sistema de ordenação e cópia implementado

    # Todas as funções PCA foram atualizadas:
    # - create_pca_results_tab(): Interface redesenhada
    # - create_pca_results_tables(): Três tabelas unificadas  
    # - execute_pca_for_results(): Usa número de componentes selecionado
    # - update_pca_results_tables(): Popula todas as tabelas
    # - populate_simple_table(): Função auxiliar para tabelas
    # - clear_pca_results(): Limpa as três tabelas
    # - _update_pca_results_from_graph(): Atualização automática

    def create_pca_results_tab(self):
        """Cria a aba de Resultados da PCA profissional com visualização unificada."""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # === Cabeçalho e Controles ===
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        
        # Título
        title_label = QLabel(GetFromLanguage("PCATabs", 0))
        title_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                padding: 12px;
                background: #ecf0f1;
                border-radius: 6px;
                border-left: 4px solid #3498db;
            }
        """)
        header_layout.addWidget(title_label)
        
        # Controles
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 15, 0, 10)
        
        # Seleção do tipo de dados
        data_type_label = QLabel(GetFromLanguage("PCATabs", 1))
        data_type_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50;")
        self.pca_data_type_combo = QComboBox()
        self.pca_data_type_combo.addItems(["Dados Originais (Índices)", "Rankings"])
        self.pca_data_type_combo.setCurrentText("Dados Originais (Índices)")
        self.pca_data_type_combo.setMinimumWidth(200)
        
        
        # Seleção do número de componentes
        components_label = QLabel(GetFromLanguage("PCATabs", 2))
        components_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50;")
        self.pca_n_components_spin = QSpinBox()
        self.pca_n_components_spin.setRange(2, 10)
        self.pca_n_components_spin.setValue(5)
        
        
        # Botão para executar PCA
        self.run_pca_button = QPushButton(GetFromLanguage("PCATabs", 3))
        self.run_pca_button.setStyleSheet("""
            QPushButton {
                background: #27ae60;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
                min-width: 150px;
            }
            QPushButton:hover {
                background: #219a52;
            }
            QPushButton:pressed {
                background: #1e8449;
            }
        """)
        self.run_pca_button.clicked.connect(self.execute_pca_for_results)
        
        # Status da PCA
        self.pca_status_label = QLabel(GetFromLanguage("PCATabs", 4 ))
        self.pca_status_label.setStyleSheet("""
            QLabel {
                color: #7f8c8d;
                font-style: italic;
                padding: 8px;
            }
        """)
        
        controls_layout.addWidget(data_type_label)
        controls_layout.addWidget(self.pca_data_type_combo)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(components_label)
        controls_layout.addWidget(self.pca_n_components_spin)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.run_pca_button)
        controls_layout.addStretch()
        controls_layout.addWidget(self.pca_status_label)
        
        header_layout.addWidget(controls_widget)
        main_layout.addWidget(header_widget)

        # === TabWidget para os tipos de resultados ===
        self.pca_results_tabs = QTabWidget()
        self.pca_results_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                background: white;
            }
            QTabWidget::tab-bar {
                alignment: left;
            }
            QTabBar::tab {
                background: #ecf0f1;
                border: 1px solid #bdc3c7;
                padding: 12px 24px;
                margin-right: 2px;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
                color: #2c3e50;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom: 2px solid white;
                color: #3498db;
            }
            QTabBar::tab:hover:!selected {
                background: #d5dbdb;
            }
        """)

        # Criar as três abas principais
        self.create_pca_results_tables()
        main_layout.addWidget(self.pca_results_tabs, 1)

        # Inicializar com dados vazios
        self.clear_pca_results()

        return tab

    def create_pca_results_tables(self):
        """Cria as tabelas unificadas para visualizar todos os componentes."""
        
        # 1. Aba de Autovalores e Variância Explicada
        eigenvalues_widget = QWidget()
        eigenvalues_layout = QVBoxLayout(eigenvalues_widget)
        eigenvalues_layout.setContentsMargins(15, 15, 15, 15)
        
        eigenvalues_title = QLabel(GetFromLanguage("PCATabs", 5))
        eigenvalues_title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
                background: #f8f9fa;
                padding: 8px;
                border-radius: 4px;
                margin-bottom: 10px;
            }
        """)
        eigenvalues_layout.addWidget(eigenvalues_title)
        
        self.pca_eigenvalues_table = QTableWidget()
        self.pca_eigenvalues_table.setAlternatingRowColors(True)
        self.pca_eigenvalues_table.setSortingEnabled(True)
        self.setup_table_copy_functionality(self.pca_eigenvalues_table, "pca_eigenvalues")
        self.pca_eigenvalues_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                gridline-color: #ecf0f1;
                background-color: white;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #ecf0f1;
            }
            QHeaderView::section {
                background: #34495e;
                color: white;
                padding: 10px;
                font-weight: bold;
                border: none;
            }
        """)
        eigenvalues_layout.addWidget(self.pca_eigenvalues_table)
        self.pca_results_tabs.addTab(eigenvalues_widget, GetFromLanguage("PCATabs", 6))
        self.pca_results_tabs.setStyleSheet("""
            QTabWidget{
               
            }
        """)
        
        ##TODO

        # 2. Aba de Cargas das Variáveis (Loadings)
        loadings_widget = QWidget()
        loadings_layout = QVBoxLayout(loadings_widget)
        loadings_layout.setContentsMargins(15, 15, 15, 15)
        
        loadings_title = QLabel(GetFromLanguage("PCATabs", 7))
        loadings_title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
                padding: 8px;
                background: #f8f9fa;
                border-radius: 4px;
                margin-bottom: 10px;
            }
        """)
        loadings_layout.addWidget(loadings_title)
        
        self.pca_loadings_table = QTableWidget()
        self.pca_loadings_table.setAlternatingRowColors(True)
        self.pca_loadings_table.setSortingEnabled(True)
        self.setup_table_copy_functionality(self.pca_loadings_table, "pca_loadings")
        self.pca_loadings_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                gridline-color: #ecf0f1;
                background-color: white;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #ecf0f1;
            }
            QHeaderView::section {
                background: #34495e;
                color: white;
                padding: 10px;
                font-weight: bold;
                border: none;
            }
        """)
        loadings_layout.addWidget(self.pca_loadings_table)
        self.pca_results_tabs.addTab(loadings_widget, GetFromLanguage("PCATabs", 8))
        
        # 3. Aba de Coordenadas dos Indivíduos
        coordinates_widget = QWidget()
        coordinates_layout = QVBoxLayout(coordinates_widget)
        coordinates_layout.setContentsMargins(15, 15, 15, 15)
        
        coordinates_title = QLabel(GetFromLanguage("PCATabs", 9))
        coordinates_title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
                padding: 8px;
                background: #f8f9fa;
                border-radius: 4px;
                margin-bottom: 10px;
            }
        """)
        coordinates_layout.addWidget(coordinates_title)
        
        self.pca_coordinates_table = QTableWidget()
        self.pca_coordinates_table.setAlternatingRowColors(True)
        self.pca_coordinates_table.setSortingEnabled(True)
        self.setup_table_copy_functionality(self.pca_coordinates_table, "pca_coordinates")
        self.pca_coordinates_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                gridline-color: #ecf0f1;
                background-color: white;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #ecf0f1;
            }
            QHeaderView::section {
                background: #34495e;
                color: white;
                padding: 10px;
                font-weight: bold;
                border: none;
            }
        """)
        coordinates_layout.addWidget(self.pca_coordinates_table)
        self.pca_results_tabs.addTab(coordinates_widget, GetFromLanguage("PCATabs", 9))

    def execute_pca_for_results(self):
        """Executa a PCA e popula as tabelas de resultados."""
        try:
            # Verificar se há dados
            if self.indices.empty:
                QMessageBox.warning(self, "Dados Ausentes", 
                                  "É necessário calcular os índices antes de executar a PCA.")
                return

            data_type = self.pca_data_type_combo.currentText()
            n_components_requested = self.pca_n_components_spin.value()
            
            if data_type == "Rankings":
                if self.rankings_df.empty:
                    QMessageBox.warning(self, "Rankings Ausentes", 
                                      "É necessário ter rankings calculados para PCA de rankings.")
                    return
                data_source = self.rankings_df
                pca_results = self.analysis.perform_pca_rankings(data_source, n_components=n_components_requested, scaling=True)
            else:
                data_source = self.indices
                pca_results = self.analysis.perform_pca(data_source, n_components=n_components_requested, scaling=True)

            # Processar resultados
            pca_df, explained_variance_ratio, loadings, feature_names, n_components = pca_results
            
            # Garantir que os nomes dos genótipos sejam preservados no DataFrame
            genotype_names = []
            possible_genotype_cols = ['Genótipo', 'Genotype', 'ID', 'Nome', 'Name', 'Genotipo']
            genotype_col_found = None
            
            for col in possible_genotype_cols:
                if col in data_source.columns:
                    genotype_col_found = col
                    genotype_names = data_source[col].tolist()
                    break
            
            if not genotype_names:
                # Fallback: usar índices ou criar nomes genéricos
                if hasattr(data_source, 'index') and not data_source.index.equals(pd.RangeIndex(len(data_source))):
                    genotype_names = data_source.index.tolist()
                else:
                    genotype_names = [f'Genótipo_{i+1}' for i in range(len(data_source))]
            
            # Aplicar os nomes ao DataFrame de coordenadas
            if len(genotype_names) == len(pca_df):
                pca_df.index = genotype_names
            elif len(genotype_names) > len(pca_df):
                pca_df.index = genotype_names[:len(pca_df)]
            else:
                extended_names = genotype_names + [f'Genótipo_{i+1}' for i in range(len(genotype_names), len(pca_df))]
                pca_df.index = extended_names
            
            # Armazenar resultados
            self.current_pca_detailed_results = {
                'coordinates': pca_df,
                'explained_variance_ratio': explained_variance_ratio,
                'loadings': loadings,
                'feature_names': feature_names,
                'n_components': n_components,
                'data_type': data_type,
                'eigenvalues': explained_variance_ratio * loadings.shape[0]  # Aproximação dos autovalores
            }
            
            # Atualizar as tabelas
            self.update_pca_results_tables()
            
            # Atualizar status
            self.pca_status_label.setText(f"Análise concluída: {n_components} componentes ({data_type})")
            self.pca_status_label.setStyleSheet("QLabel { color: #27ae60; font-weight: bold; }")

        except Exception as e:
            QMessageBox.critical(self, "Erro na PCA", f"Erro ao executar PCA:\n{str(e)}")
            self.pca_status_label.setText("Erro na análise - verifique os dados")
            self.pca_status_label.setStyleSheet("QLabel { color: #e74c3c; font-weight: bold; }")

    def update_pca_results_tables(self):
        """Atualiza todas as tabelas com os resultados da PCA."""
        if not hasattr(self, 'current_pca_detailed_results'):
            return
            
        results = self.current_pca_detailed_results
        
        # 1. Tabela de Autovalores e Variância Explicada (todos os componentes)
        eigenvalues_data = []
        eigenvalues_data.append(['Componente', 'Autovalor', 'Variância Explicada (%)', 'Variância Acumulada (%)'])
        
        cumulative_variance = 0
        for i in range(results['n_components']):
            eigenvalue = results['eigenvalues'][i]
            variance_pct = results['explained_variance_ratio'][i] * 100
            cumulative_variance += variance_pct
            
            eigenvalues_data.append([
                f'PC{i+1}',
                f'{eigenvalue:.6f}',
                f'{variance_pct:.4f}%',
                f'{cumulative_variance:.4f}%'
            ])
        
        self.populate_simple_table(self.pca_eigenvalues_table, eigenvalues_data)
        
        # 2. Tabela de Cargas das Variáveis (todos os componentes)
        loadings_data = []
        header = ['Variável'] + [f'PC{i+1}' for i in range(results['n_components'])]
        loadings_data.append(header)
        
        for j, variable in enumerate(results['feature_names']):
            row = [variable]
            for i in range(results['n_components']):
                row.append(f"{results['loadings'][j][i]:.6f}")
            loadings_data.append(row)
        
        self.populate_simple_table(self.pca_loadings_table, loadings_data)
        
        # 3. Tabela de Coordenadas dos Indivíduos (todos os componentes)
        coordinates_data = []
        header = ['Indivíduo'] + [f'PC{i+1}' for i in range(results['n_components'])]
        coordinates_data.append(header)
        
        # Usar nomes dos genótipos diretamente do índice das coordenadas
        coordinates = results['coordinates']
        
        print(f"DEBUG Coords: Tipo de coordinates: {type(coordinates)}")
        print(f"DEBUG Coords: Colunas: {list(coordinates.columns) if hasattr(coordinates, 'columns') else 'N/A'}")
        print(f"DEBUG Coords: Índice: {coordinates.index.tolist()[:5] if hasattr(coordinates, 'index') else 'N/A'}... (primeiros 5)")
        print(f"DEBUG Coords: Shape: {coordinates.shape if hasattr(coordinates, 'shape') else 'N/A'}")
        
        for i in range(len(coordinates)):
            # Usar o índice das coordenadas que já contém os nomes reais dos genótipos
            if hasattr(coordinates, 'index') and len(coordinates.index) > i:
                individual_name = str(coordinates.index[i])
                print(f"DEBUG Row {i}: Nome do genótipo obtido do índice: '{individual_name}'")
            else:
                individual_name = f'Genótipo_{i+1}'
                print(f"DEBUG Row {i}: Nome genérico usado: '{individual_name}'")
            
            row = [individual_name]
            for j in range(results['n_components']):
                if hasattr(coordinates, 'iloc'):
                    coord_value = coordinates.iloc[i, j]
                else:
                    coord_value = coordinates[i, j] if len(coordinates.shape) > 1 else coordinates[i]
                row.append(f"{coord_value:.6f}")
            coordinates_data.append(row)
            
            # Debug apenas das primeiras 3 linhas para não poluir
            if i < 3:
                print(f"DEBUG Row {i} final: {row}")
        
        print(f"DEBUG: Dados finais para a tabela (primeiras 3 linhas): {coordinates_data[:4]}")
        self.populate_simple_table(self.pca_coordinates_table, coordinates_data)

    def populate_simple_table(self, table_widget, data):
        """Popula uma tabela com dados simples (lista de listas)."""
        if not data:
            table_widget.clear()
            return
            
        table_widget.clear()
        table_widget.setRowCount(len(data))
        table_widget.setColumnCount(len(data[0]))
        
        # Cabeçalho
        if len(data) > 1:
            table_widget.setHorizontalHeaderLabels(data[0])
            data_rows = data[1:]
            table_widget.setRowCount(len(data_rows))
        else:
            data_rows = data
            
        # Dados
        for i, row in enumerate(data_rows):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                
                # Adicionar valor numérico para ordenação se for número
                try:
                    numeric_value = float(value.replace('%', ''))
                    item.setData(Qt.UserRole, numeric_value)
                except:
                    pass
                    
                table_widget.setItem(i, j, item)
        
        table_widget.resizeColumnsToContents()

    def clear_pca_results(self):
        """Limpa todas as tabelas de resultados da PCA."""
        self.pca_eigenvalues_table.clear()
        self.pca_loadings_table.clear()
        self.pca_coordinates_table.clear()

    def apply_smart_column_resize(self, table_widget):
        """Aplica redimensionamento inteligente das colunas baseado no conteúdo."""
        try:
            header = table_widget.horizontalHeader()
            col_count = table_widget.columnCount()
            
            if col_count == 0:
                return
                
            if col_count <= 4:
                # Poucas colunas: esticar para preencher a largura
                header.setSectionResizeMode(QHeaderView.Stretch)
            elif col_count <= 8:
                # Colunas médias: redimensionar para conteúdo com mínimo
                header.setSectionResizeMode(QHeaderView.ResizeToContents)
                header.setMinimumSectionSize(70)
                # Permitir que a primeira coluna seja um pouco maior
                if col_count > 1:
                    header.setSectionResizeMode(0, QHeaderView.Interactive)
                    table_widget.setColumnWidth(0, 120)
            else:
                # Muitas colunas: scroll horizontal
                header.setSectionResizeMode(QHeaderView.Interactive)
                header.setDefaultSectionSize(80)
                header.setMinimumSectionSize(60)
                # Primeira coluna mais larga para rótulos
                table_widget.setColumnWidth(0, 100)
        except Exception as e:
            print(f"Warning: Could not apply smart column resize: {e}")

    def populate_simple_table(self, table_widget, data):
        """Popula uma tabela com dados simples (lista de listas)."""
        if not data:
            table_widget.clear()
            return
            
        table_widget.clear()
        table_widget.setRowCount(len(data))
        table_widget.setColumnCount(len(data[0]))
        
        # Cabeçalho
        if len(data) > 1:
            table_widget.setHorizontalHeaderLabels(data[0])
            data_rows = data[1:]
            table_widget.setRowCount(len(data_rows))
        else:
            data_rows = data
            
        # Dados
        for i, row in enumerate(data_rows):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                
                # Adicionar valor numérico para ordenação se for número
                try:
                    # Se não for a primeira coluna (que é texto - genótipo)
                    if j > 0:
                        numeric_value = float(str(value).replace('%', ''))
                        item.setData(Qt.UserRole, numeric_value)
                except:
                    pass
                    
                table_widget.setItem(i, j, item)
        
        # Ajustar automaticamente o tamanho das colunas
        table_widget.resizeColumnsToContents()
        
        # Aplicar redimensionamento inteligente
        self.apply_smart_column_resize(table_widget)

    def clear_pca_results(self):
        """Limpa todas as tabelas de resultados da PCA."""
        self.pca_eigenvalues_table.clear()
        self.pca_loadings_table.clear()
        self.pca_coordinates_table.clear()

    def _update_pca_results_from_graph(self, data_source, use_rankings):
        """Atualiza a aba de Resultados da PCA quando um gráfico PCA é gerado."""
        try:
            # Executar PCA com o número de componentes selecionado na aba de resultados
            n_components = self.pca_n_components_spin.value() if hasattr(self, 'pca_n_components_spin') else 5
            
            if use_rankings:
                pca_results = self.analysis.perform_pca_rankings(data_source, n_components=n_components, scaling=True)
                data_type = "Rankings"
                # Atualizar o combo box para refletir o tipo de dados
                self.pca_data_type_combo.setCurrentText("Rankings")
            else:
                pca_results = self.analysis.perform_pca(data_source, n_components=n_components, scaling=True)
                data_type = "Dados Originais (Índices)"
                # Atualizar o combo box para refletir o tipo de dados
                self.pca_data_type_combo.setCurrentText("Dados Originais (Índices)")

            # Processar resultados
            pca_df, explained_variance_ratio, loadings, feature_names, n_components = pca_results
            
            # Garantir que os nomes dos genótipos sejam preservados no DataFrame
            genotype_names = []
            possible_genotype_cols = ['Genótipo', 'Genotype', 'ID', 'Nome', 'Name', 'Genotipo']
            genotype_col_found = None
            
            for col in possible_genotype_cols:
                if col in data_source.columns:
                    genotype_col_found = col
                    genotype_names = data_source[col].tolist()
                    break
            
            if not genotype_names:
                # Fallback: usar índices ou criar nomes genéricos
                if hasattr(data_source, 'index') and not data_source.index.equals(pd.RangeIndex(len(data_source))):
                    genotype_names = data_source.index.tolist()
                else:
                    genotype_names = [f'Genótipo_{i+1}' for i in range(len(data_source))]
            
            # Aplicar os nomes ao DataFrame de coordenadas
            if len(genotype_names) == len(pca_df):
                pca_df.index = genotype_names
            elif len(genotype_names) > len(pca_df):
                pca_df.index = genotype_names[:len(pca_df)]
            else:
                extended_names = genotype_names + [f'Genótipo_{i+1}' for i in range(len(genotype_names), len(pca_df))]
                pca_df.index = extended_names
            
            # Armazenar resultados
            self.current_pca_detailed_results = {
                'coordinates': pca_df,
                'explained_variance_ratio': explained_variance_ratio,
                'loadings': loadings,
                'feature_names': feature_names,
                'n_components': n_components,
                'data_type': data_type,
                'eigenvalues': explained_variance_ratio * loadings.shape[0]  # Aproximação dos autovalores
            }
            
            # Atualizar as tabelas
            self.update_pca_results_tables()
            
            # Atualizar status
            self.pca_status_label.setText(f"PCA atualizada automaticamente com {n_components} componentes ({data_type})")
            self.pca_status_label.setStyleSheet("QLabel { color: #27ae60; font-weight: bold; }")

        except Exception as e:
            print(f"Aviso: Erro ao atualizar resultados da PCA automaticamente: {e}")
            # Não mostrar erro ao usuário, apenas logar
            self.pca_status_label.setText("PCA automática não disponível - use o botão 'Executar PCA'")
            self.pca_status_label.setStyleSheet("QLabel { color: #ff9800; font-style: italic; }")

    def create_help_tab(self):
        """Creates the 'Help/About' tab with useful information."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("font-size: 16px;")
        text_edit.setHtml("""
            <h1 style='font-size: 2.2em; color: #2196F3;'><span style='vertical-align: middle;'>🌱</span> iPASTIC Desktop v2.5</h1>
            <p>Esta aplicação foi projetada para análise detalhada de dados de estresse em plantas, incluindo cálculos, estatísticas, gráficos e exportação de resultados.</p>

            <h2 style='color: #4CAF50;'>Como Usar</h2>
            <ol>
                <li><b>Importar Dados:</b> Importe arquivos <code>.csv</code> ou <code>.xlsx</code> com dados de produtividade. O sistema aceita 2 ou 3 colunas (<code>Genotipo</code>, <code>Yp</code>, <code>Ys</code>).</li>
                <li><b>Editar Dados:</b> Adicione, remova ou limpe linhas/células facilmente. Use atalhos (<code>Ctrl++</code>, <code>Ctrl+-</code>, <code>Delete</code>) ou o menu de contexto (clique direito) para copiar, colar, limpar células e adicionar/remover linhas. Copie e cole dados diretamente do Excel.</li>
                <li><b>Copiar/Colar:</b> Selecione células e use <b>Ctrl+C</b> para copiar e <b>Ctrl+V</b> para colar. O app aceita colagem de múltiplas células e ajusta o tamanho da tabela automaticamente.</li>
                <li><b>Calcular Índices:</b> Clique em <b>Calcular Índices</b> para gerar todos os índices de tolerância ao estresse. Os resultados aparecem nas abas "Índices" e "Rankings".</li>
                <li><b>Rankings:</b> A aba "Rankings" mostra a classificação dos genótipos para cada índice, facilitando a seleção dos mais tolerantes.</li>
                <li><b>Análises Estatísticas:</b> A aba "Análises Estatísticas" oferece:
                    <ul>
                        <li><b>Estatísticas Descritivas:</b> Média, desvio padrão, mínimo, máximo e outros para cada índice.</li>
                        <li><b>Matriz de Correlação:</b> Relação entre índices usando métodos Pearson, Spearman ou Kendall. Cores indicam força da correlação.</li>
                        <li><b>Testes de Normalidade:</b> Resultados dos testes Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling, etc. Mostra p-valores e interpretações. Colunas sem p-valor exibem 'N/A' ou explicação.</li>
                        <li><b>Detecção de Outliers:</b> Identifica genótipos com valores extremos em cada índice.</li>
                        <li><b>Análise de Clusters:</b> Agrupa genótipos por similaridade usando K-Means ou Agglomerativo. Exibe número de clusters, qualidade (Silhouette Score) e tamanho de cada grupo. O resultado é mostrado em tabela e pode ser usado para colorir gráficos.</li>
                    </ul>
                </li>
                <li><b>Gráficos:</b> Gere gráficos interativos:
                    <ul>
                        <li><b>Dispersão (Yp x Ys):</b> Visualiza relação entre produtividade com e sem estresse.</li>
                        <li><b>PCA Biplot:</b> Reduz dimensionalidade e mostra agrupamentos e variáveis mais influentes. Permite escolher matriz de covariância/correlação e aplicar normalização.</li>
                        <li><b>Dendrograma:</b> Exibe hierarquia de agrupamento dos genótipos com opções de método e métrica.</li>
                        <li><b>Gráfico de Barras/Histograma:</b> Analisa distribuição de qualquer índice. O eixo X nunca será 'Cluster'.</li>
                        <li><b>Heatmap de Correlação:</b> Visualiza todas as correlações entre índices, com escolha do método.</li>
                        <li>Personalize cores, rótulos, agrupamento e exporte os gráficos em vários formatos. Os gráficos não são incluídos nos relatórios, mas podem ser exportados individualmente.</li>
                    </ul>
                </li>
                <li><b>Exportar Resultados:</b> Salve todos os dados, índices e rankings em Excel (<code>.xlsx</code>) ou exporte gráficos como imagem (PNG, PDF, SVG, JPEG).</li>
                <li><b>Relatórios:</b> Gere relatórios completos em Excel, incluindo estatísticas, interpretações automáticas e observações. Gráficos não são incluídos nos relatórios.</li>
                <li><b>Regressão Múltipla:</b> Analise relações lineares entre os índices de tolerância ao estresse:
                    <ul>
                        <li><b>Seleção de Variáveis:</b> Escolha uma variável dependente (Y) e múltiplas independentes (X) entre os índices calculados.</li>
                        <li><b>Opções Avançadas:</b> Inclua/exclua intercepto, padronize variáveis e ative análise de resíduos.</li>
                        <li><b>Resultados Completos:</b> R², coeficientes, testes de significância, diagnósticos (Breusch-Pagan, Jarque-Bera, Durbin-Watson), VIF para multicolinearidade.</li>
                        <li><b>Interpretação Automática:</b> Análise de resíduos, detecção de outliers e recomendações para melhorar o modelo.</li>
                    </ul>
                </li>
                <li><b>Visual:</b> Interface moderna, fonte ampliada, ícones e cores para facilitar a navegação.</li>
            </ol>

            <h2 style='color: #FF9800;'>Exemplo Prático</h2>
            <table border='1' cellpadding='4' style='border-collapse:collapse;'>
                <tr><th>Genotipo</th><th>Yp</th><th>Ys</th></tr>
                <tr><td>A</td><td>5000</td><td>4500</td></tr>
                <tr><td>B</td><td>5200</td><td>3900</td></tr>
            </table>
            <p>Para o genótipo A: <b>RP = (5000 - 4500) / 5000 × 100 = 10%</b> (alta tolerância).</p>

            <h2 style='color: #009688;'>Dicas Rápidas</h2>
            <ul>
                <li>Use <b>Ctrl+C</b> e <b>Ctrl+V</b> para copiar/colar dados entre Excel e o app.</li>
                <li>Menu de contexto: clique direito na tabela para acessar opções rápidas.</li>
                <li>Exportação: gráficos podem ser salvos em PNG, PDF, SVG ou JPEG.</li>
                <li>Atalhos: <b>Ctrl++</b> (adicionar linha), <b>Ctrl+-</b> (remover linha), <b>Delete</b> (limpar célula).</li>
            </ul>

            <h2 style='color: #673AB7;'>Glossário</h2>
            <ul>
                <li><b>Tolerância:</b> Capacidade de manter produtividade sob estresse.</li>
                <li><b>Susceptibilidade:</b> Redução acentuada da produtividade sob estresse.</li>
                <li><b>Cluster:</b> Grupo de genótipos similares.</li>
                <li><b>Silhouette Score:</b> Métrica de qualidade da clusterização (0-1, melhor > 0.5).</li>
            </ul>

            <h2 style='color: #E91E63;'>FAQ - Perguntas Frequentes</h2>
            <ul>
                <li><b>Por que meu gráfico não aparece?</b> Verifique se os índices foram calculados e se há dados suficientes.</li>
                <li><b>Como corrigir erro de importação?</b> Certifique-se que o arquivo tem as colunas corretas e está salvo como .csv ou .xlsx.</li>
                <li><b>Como exportar resultados?</b> Use o botão de exportação nas abas de gráficos ou relatórios.</li>
                <li><b>O que significa p-valor 'N/A'?</b> Indica que o teste não pôde ser realizado para aquele índice.</li>
            </ul>

            <h2 style='color: #00BCD4;'>Links Úteis</h2>
            <ul>
                <li><a href='https://www.ufersa.edu.br/portal/laboratorio-de-estresse-em-plantas/' target='_blank'>Laboratório de Estresse em Plantas</a></li>
                <li><a href='https://github.com/stefeson/ipastic' target='_blank'>Repositório GitHub</a></li>
                <li><a href='mailto:stefeson@ufersa.edu.br'>Contato/Suporte</a></li>
            </ul>

            <h2 style='color: #607D8B;'>Mensagens de Erro Comuns</h2>
            <ul>
                <li><b>Erro de Importação:</b> Arquivo inválido ou colunas ausentes. Solução: revise o arquivo.</li>
                <li><b>Dados Ausentes:</b> Nenhum dado carregado. Solução: importe um arquivo válido.</li>
                <li><b>Erro ao Gerar Gráfico:</b> Dados insuficientes ou índices não calculados.</li>
            </ul>

            <h2 style='color: #3F51B5;'>Descrição dos Índices e Fórmulas Originais</h2>
            <ul>
                <li><b>RP:</b> Redução Relativa na Produtividade<br><span style='font-size:1.1em;'>RP = (Yp - Ys) / Yp × 100</span><br>Indica a porcentagem de redução na produtividade sob estresse. Quanto menor, mais tolerante.</li>
                <li><b>TOL:</b> Índice de Tolerância<br><span style='font-size:1.1em;'>TOL = YP - YS</span><br>Diferença absoluta entre produtividade sem e com estresse. Valores baixos indicam maior tolerância.</li>
                <li><b>MP:</b> Média da Produtividade<br><span style='font-size:1.1em;'>MP = (YP + YS) / 2</span><br>Média simples das produtividades. Útil para comparar genótipos em ambos os ambientes.</li>
                <li><b>GMP:</b> Média Geométrica da Produtividade<br><span style='font-size:1.1em;'>GMP = sqrt(YS × YP)</span><br>Valor intermediário entre YP e YS, menos sensível a extremos.</li>
                <li><b>HMS:</b> Média Harmônica da Produtividade<br><span style='font-size:1.1em;'>HMS = 2 × YS × YP / (YS + YP)</span><br>Destaca genótipos com desempenho equilibrado nos dois ambientes.</li>
                <li><b>SSI:</b> Índice de Suscetibilidade ao Estresse<br><span style='font-size:1.1em;'>SSI = [1 - (YS / YP)] / [1 - (média YS / média YP)]</span><br>Compara a redução relativa de cada genótipo com a média geral. Valores baixos indicam tolerância.</li>
                <li><b>STI:</b> Índice de Tolerância ao Estresse<br><span style='font-size:1.1em;'>STI = (YS × YP) / (média YP)^2</span><br>Identifica genótipos produtivos em ambos os ambientes.</li>
                <li><b>YI:</b> Índice de Produtividade<br><span style='font-size:1.1em;'>YI = YS / média YS</span><br>Compara a produtividade sob estresse com a média geral.</li>
                <li><b>YSI:</b> Índice de Estabilidade da Produtividade<br><span style='font-size:1.1em;'>YSI = YS / YP</span><br>Proporção entre produtividades, indica estabilidade sob estresse.</li>
                <li><b>RSI:</b> Índice Relativo de Estresse<br><span style='font-size:1.1em;'>RSI = (YS / YP) × (média YS / média YP)</span><br>Compara desempenho relativo do genótipo com o desempenho médio.</li>
            </ul>
            <hr>
            <p style='font-size:1.1em;'>Desenvolvido por: Laboratório de Estresse em Plantas.<br>Suporte: <a href='mailto:stefeson@ufersa.edu.br'>stefeson@ufersa.edu.br</a></p>
        """)
        layout.addWidget(text_edit)
        return tab

    # --- Core Application Logic ---

    def import_data(self, file_path=None):
        """Handles the data import process, including file dialog and validation."""
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(self, "Importar Arquivo de Dados", "", "Arquivos de Dados (*.xlsx *.xls *.csv *.txt)")
        if not file_path:
            return

        try:
            self.clear_all_data() # Clear previous data before importing new
            df = self.data_manager.import_file(file_path)
            self.data, self.original_column_names = self.data_manager.validate_and_standardize(df)
            self.populate_input_table()
            self.status_bar.showMessage(f"Dados importados com sucesso de: {os.path.basename(file_path)}", 5000)
        except (ValueError, FileNotFoundError) as e:
            QMessageBox.critical(self, "Erro de Importação", str(e))
            self.clear_all_data()
        finally:
            self.update_ui_state()

    def clear_all_data(self):
        """Resets all data and UI elements to their initial state."""
        self.data = pd.DataFrame()
        self.indices = pd.DataFrame()
        self.rankings_df = pd.DataFrame()
        self.original_column_names = [GetFromLanguage("Headers", i) for i in range(3)]

        # Clear tables
        self.input_table.clearContents(); self.input_table.setRowCount(20); self.input_table.setHorizontalHeaderLabels(self.original_column_names)
        self.indices_table.clearContents(); self.indices_table.setRowCount(0); self.indices_table.setColumnCount(0)
        self.rank_table.clearContents(); self.rank_table.setRowCount(0); self.rank_table.setColumnCount(0)

        # Clear statistics tables
        if hasattr(self, 'descriptive_table'):
            self.clear_table(self.descriptive_table)
        if hasattr(self, 'correlation_table'):
            self.clear_table(self.correlation_table)
        if hasattr(self, 'normality_table'):
            self.clear_table(self.normality_table)
        if hasattr(self, 'outliers_table'):
            self.clear_table(self.outliers_table)
        if hasattr(self, 'cluster_analysis_table'):
            self.clear_table(self.cluster_analysis_table)
        
        # Clear enhanced analysis tables (NOVO)
        if hasattr(self, 'heritability_table'):
            self.clear_table(self.heritability_table)
        # Limpeza da tabela de estabilidade removida

        # Clear graph
        if self.current_graph_fig:
            plt.close(self.current_graph_fig)
            self.current_graph_fig = None
        
        # Limpar widgets do layout de forma mais segura
        while self.graph_preview_layout.count():
            child = self.graph_preview_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Adicionar placeholder de volta
        placeholder = QLabel("Visualização de Gráficos\n\nSelecione um tipo de gráfico e clique em 'Gerar Gráfico' para visualizar")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("""
            color: #666666;
            font-size: 14px;
            padding: 60px 20px;
            font-weight: 500;
        """)
        self.graph_preview_layout.addWidget(placeholder)

        self.status_bar.showMessage("Dados limpos. Pronto para importar um novo arquivo.", 5000)
        self.update_ui_state()

    def change_language(self):
        QMessageBox.warning(self, GetFromLanguage("LanguageChangePopup", 0), GetFromLanguage("LanguageChangePopup", 1))
        print("DEBUG: change_language called")
        global LANGUAGE_CODE
        Language_config_file = os.path.join(os.path.dirname(__file__), 'language_config.json')
        try:
            with open(Language_config_file, 'r', encoding='utf-8') as f:
                print(f.read())
            with open(Language_config_file, 'w') as f:        
                global LANGUAGE_CODE
                if(LANGUAGE_CODE == "EN"):
                    f.write("""{\n    "LanguageCode": "PTBR"\n}""")
                    print("DEBUG: Writing language code 'a' to config file")
                else:
                    print("DEBUG: Writing language code 'b' to config file")
                    f.write("""{\n    "LanguageCode": "EN"\n}""")
        except Exception as e:
            print("DEBUG: Error writing language config:", e)
        
        self.update_ui_state()
        return LANGUAGE_CODE


    #TODO:INVESTIGATE IF SEPARETA TREAD FUNCTION CALLED
    def start_calculation(self):
        """Initiates the index calculation in a worker thread."""
        self.data = self.get_data_from_table()
        if self.data.empty:
            QMessageBox.warning(self, "Dados Ausentes", "A tabela de entrada está vazia ou inválida. Por favor, importe dados primeiro.")
            return
        self.run_in_worker(self.analysis.calculate_indices, self.data, self.classification_limits, on_finish=self.on_calculation_finished)

    def on_calculation_finished(self, result):
        """Handles the results from the calculation worker."""
        try:
            self.indices = result
            self.rankings_df = self.analysis.generate_rankings_df(self.indices)

            # Populate UI
            self.populate_indices_table()
            self.populate_rankings_table()
            
            # Update graph controls using the new centralized function
            self.update_variable_lists()
            
            # Executar automaticamente análises estatísticas após calcular índices
            self.refresh_statistical_analyses()

            self.status_bar.showMessage("Cálculos e análises estatísticas concluídos!", 5000)
            self.central_widget.setCurrentIndex(1) # Switch to indices tab
            self.update_ui_state()
            
        except Exception as e:
            QMessageBox.critical(self, "Erro no Cálculo", f"Erro ao processar resultados dos cálculos:\n{str(e)}")
            logging.error(f"Error in on_calculation_finished: {e}", exc_info=True)

    def update_variable_lists(self):
        """Populates the variable selection widgets based on whether ranking data is used."""
        use_rankings = self.use_rankings_graph.isChecked()
        
        source_df = self.rankings_df if use_rankings else self.indices
        
        if source_df is None or source_df.empty:
            return

        numeric_cols = source_df.select_dtypes(include=np.number).columns.tolist()
        
        # Clear all lists before populating
        self.single_var_combo.clear()
        self.multi_var_list_available.clear()
        self.multi_var_list_selected.clear()
        self.scatter_x_combo.clear()
        self.scatter_y_combo.clear()
        self.graph_3d_x_combo.clear()
        self.graph_3d_y_combo.clear()
        self.graph_3d_z_combo.clear()

        # Populate all lists with new columns
        self.single_var_combo.addItems(numeric_cols)
        self.multi_var_list_available.addItems(numeric_cols)
        self.scatter_x_combo.addItems(numeric_cols)
        self.scatter_y_combo.addItems(numeric_cols)
        self.graph_3d_x_combo.addItems(numeric_cols)
        self.graph_3d_y_combo.addItems(numeric_cols)
        self.graph_3d_z_combo.addItems(numeric_cols)

        # --- Set sensible defaults to provide an initial graph ---
        # Default for scatter plot
        if self.scatter_x_combo.count() > 0:
            self.scatter_x_combo.setCurrentIndex(0)
        if self.scatter_y_combo.count() > 1:
            self.scatter_y_combo.setCurrentIndex(1)
        
        # Default for 3D plot
        if self.graph_3d_x_combo.count() > 0:
            # Set X to first variable (usually Yp)
            self.graph_3d_x_combo.setCurrentIndex(0)
        
        if self.graph_3d_y_combo.count() > 0:
            # Try to find "Ys" for Y axis, otherwise use second available
            ys_index = -1
            for i in range(self.graph_3d_y_combo.count()):
                if self.graph_3d_y_combo.itemText(i) == 'Ys':
                    ys_index = i
                    break
            
            if ys_index >= 0:
                self.graph_3d_y_combo.setCurrentIndex(ys_index)
            elif self.graph_3d_y_combo.count() > 1:
                self.graph_3d_y_combo.setCurrentIndex(1)
        
        if self.graph_3d_z_combo.count() > 2:
            self.graph_3d_z_combo.setCurrentIndex(2)
        
        # Default for multi-selection lists (move first 3 items)
        if self.multi_var_list_available.count() > 0:
            for i in range(min(3, self.multi_var_list_available.count())):
                self.multi_var_list_selected.addItem(self.multi_var_list_available.takeItem(0))
        
        self._handle_auto_update()

    
    def start_clustering_with_options(self):
        """Initiates clustering analysis considering user options for data type."""
        if self.indices.empty:
            QMessageBox.warning(self, "Dados Ausentes", "É necessário calcular os índices antes de executar o clustering.")
            return
            
        use_rankings = self.stats_use_rankings.isChecked() if hasattr(self, 'stats_use_rankings') else False
        
        # Get clustering parameters
        method = 'kmeans' if self.cluster_method_combo.currentText() == 'K-Means' else 'agglomerative'
        n_clusters = 'auto' if self.cluster_number_combo.currentText() == 'Automático' else int(self.cluster_number_combo.currentText())
        max_clusters = 10  # Default since max_clusters_spin was removed
        
        if use_rankings:
            if self.rankings_df.empty:
                QMessageBox.warning(self, "Rankings Ausentes", "É necessário calcular os rankings antes de usar esta opção.")
                return
            self.run_in_worker(self.analysis.perform_clustering_rankings, self.rankings_df, 
                             method=method, n_clusters=n_clusters, max_clusters=max_clusters,
                             on_finish=lambda result: self.on_clustering_finished(result, use_rankings=True))
        else:
            self.run_in_worker(self.analysis.perform_clustering, self.indices, 
                             method=method, n_clusters=n_clusters, max_clusters=max_clusters,
                             on_finish=lambda result: self.on_clustering_finished(result, use_rankings=False))

    def refresh_statistical_analyses(self):
        """Refreshes all statistical analyses tables."""
        if self.indices.empty:
            QMessageBox.warning(self, "Dados Ausentes", "É necessário calcular os índices antes de executar análises estatísticas.")
            return
        
        try:
            use_rankings = self.stats_use_rankings.isChecked()
            corr_method = self.correlation_method_stats.currentText()
            
            # Choose data source
            if use_rankings and not self.rankings_df.empty:
                data_source = self.rankings_df.filter(regex='^Rank_')
                data_name = "Rankings"
            else:
                data_source = self.indices.select_dtypes(include=np.number)
                data_name = "Índices"
            
            # Update descriptive statistics
            desc_stats = self.analysis.calculate_descriptive_statistics(data_source)
            if not desc_stats.empty:
                self.populate_statistics_table(self.descriptive_table, desc_stats, 
                                             f"Estatísticas Descritivas - {data_name}")
            
            # Update correlation matrix
            if not data_source.empty:
                corr_matrix = data_source.corr(method=corr_method)
                self.populate_correlation_table(self.correlation_table, corr_matrix, 
                                              f"Correlação {corr_method.title()} - {data_name}")
            
            # Update normality tests
            normality_results = self.analysis.perform_normality_tests(data_source)
            if not normality_results.empty:
                self.populate_generic_table(self.normality_table, normality_results, 
                                          f"Testes de Normalidade - {data_name}")
            else:
                self.clear_table(self.normality_table)
            
            # Update outliers detection
            try:
                outliers_results = self.analysis.detect_outliers(self.indices)
                if not outliers_results.empty:
                    self.populate_generic_table(self.outliers_table, outliers_results, "Outliers (Z-score/IQR)")
                else:
                    self.clear_table(self.outliers_table)
            except Exception as e:
                logging.warning(f"Outlier detection failed: {e}")
                self.clear_table(self.outliers_table)

            # MFV Analysis
            try:
                self.mfv_df = self.analysis.calculate_mfv(self.indices) # Always use indices, not rankings
                if not self.mfv_df.empty:
                    self.populate_generic_table(self.mfv_table, self.mfv_df, "Análise de Pertinência (MFV)")
                else:
                    self.clear_table(self.mfv_table)
                    self.mfv_df = None
            except Exception as e:
                logging.error(f"Error during MFV analysis: {e}", exc_info=True)
                self.clear_table(self.mfv_table)
                self.mfv_df = None
                QMessageBox.warning(self, "Erro na Análise MFV", f"Não foi possível realizar a análise MFV: {e}")
            
            # Update enhanced analyses (NOVO - do artigo Eruca sativa)
            if self.enhanced_analysis and not self.indices.empty:
                try:
                    # Heritability Analysis
                    heritability_results = self.enhanced_analysis.perform_heritability_analysis(self.indices)
                    if not heritability_results.empty:
                        self.populate_generic_table(self.heritability_table, heritability_results, "Análise de Herdabilidade")
                    
                    # Análise de Estabilidade removida conforme solicitado
                        
                except Exception as e:
                    logging.error(f"Error in enhanced analyses: {e}")
            
            
            
            self.status_bar.showMessage("Análises estatísticas atualizadas!", 3000)
            
        except Exception as e:
            QMessageBox.critical(self, "Erro nas Análises", f"Erro ao calcular análises estatísticas:\n{str(e)}")
            logging.error(f"Statistical analysis error: {e}", exc_info=True)

    def on_clustering_finished(self, result, use_rankings=False):
        """Handles the results from the clustering worker."""
        try:
            clusters, n_clusters, sil_score = result
            self.indices['Cluster'] = clusters
            
            # Also add cluster to rankings if available
            if not self.rankings_df.empty:
                self.rankings_df['Cluster'] = clusters
            
            # Create cluster summary for display
            cluster_summary = []
            cluster_summary.append({
                'Métrica': 'Número de Clusters',
                'Valor': str(n_clusters),
                'Descrição': 'Número total de clusters identificados'
            })
            cluster_summary.append({
                'Métrica': 'Silhouette Score',
                'Valor': f"{sil_score:.3f}" if sil_score is not None else "N/A",
                'Descrição': 'Qualidade da clusterização (0-1, melhor > 0.5)'
            })
            
            # Cluster sizes and genotypes
            unique, counts = np.unique(clusters, return_counts=True)
            for cluster_id, size in zip(unique, counts):
                cluster_summary.append({
                    'Métrica': f'Cluster {cluster_id}',
                    'Valor': f"{size} genótipos",
                    'Descrição': f'Tamanho do cluster {cluster_id}'
                })
                # Get genotypes for the current cluster
                genotype_indices = np.where(clusters == cluster_id)[0]
                genotypes = self.indices.iloc[genotype_indices]['Genotipo'].tolist()
                # Add genotypes to the summary
                cluster_summary.append({
                    'Métrica': f'Genótipos Cluster {cluster_id}',
                    'Valor': ", ".join(genotypes),
                    'Descrição': f'Genótipos pertencentes ao cluster {cluster_id}'
                })
            
            # Convert to DataFrame and populate table
            summary_df = pd.DataFrame(cluster_summary)
            self.populate_generic_table(self.cluster_analysis_table, summary_df, "Resumo da Análise de Clusters")
            
            self.status_bar.showMessage(f"Clustering concluído! {n_clusters} clusters identificados.", 5000)

            # Automatically generate cluster mean bar plot if variables are selected
            if hasattr(self, 'cluster_plot_list_selected') and self.cluster_plot_list_selected.count() > 0:
                self._generate_cluster_mean_barplot()
            
        except Exception as e:
            QMessageBox.critical(self, "Erro no Clustering", f"Erro ao processar resultados do clustering:\n{str(e)}")
            logging.error(f"Error updating cluster analysis table: {e}", exc_info=True)
            self.clear_table(self.cluster_analysis_table)

    def _clear_layout(self, layout):
        """Recursively clear a layout and all its children."""
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().setParent(None)
                    child.widget().deleteLater()
                elif child.layout():
                    self._clear_layout(child.layout())

    def _validate_data_for_plot(self, data, plot_type):
        """Validate data before plotting."""
        if data is None or data.empty:
            raise ValueError("Dados vazios ou nulos")
        
        numeric_data = data.select_dtypes(include=np.number)
        
        validation_rules = {
            'Gráfico de Dispersão': {
                'min_numeric': 2,
                'min_rows': 2,
                'message': 'Dispersão requer pelo menos 2 variáveis numéricas e 2 observações'
            },
            'Gráfico 3D': {
                'min_numeric': 3,
                'min_rows': 3,
                'message': 'Gráfico 3D requer pelo menos 3 variáveis numéricas e 3 observações'
            },
            'PCA Biplot': {
                'min_numeric': 2,
                'min_rows': 3,
                'message': 'PCA requer pelo menos 2 variáveis numéricas e 3 observações'
            },
            'Dendrograma': {
                'min_numeric': 1,
                'min_rows': 2,
                'message': 'Dendrograma requer pelo menos 1 variável numérica e 2 observações'
            },
            'Gráfico de Barras': {
                'min_numeric': 1,
                'min_rows': 1,
                'message': 'Gráfico de barras requer pelo menos 1 variável numérica'
            },
            'Histograma': {
                'min_numeric': 1,
                'min_rows': 2,
                'message': 'Histograma requer pelo menos 1 variável numérica e 2 observações'
            },
            'Box Plot': {
                'min_numeric': 1,
                'min_rows': 2,
                'message': 'Box plot requer pelo menos 1 variável numérica e 2 observações'
            },
            'Heatmap de Correlação': {
                'min_numeric': 2,
                'min_rows': 2,
                'message': 'Heatmap requer pelo menos 2 variáveis numéricas e 2 observações'
            }
        }
        
        if plot_type not in validation_rules:
            return True  # Unknown plot type, assume valid
            
        rules = validation_rules[plot_type]
        
        # Check minimum numeric columns
        if len(numeric_data.columns) < rules['min_numeric']:
            raise ValueError(f"{rules['message']} - {len(numeric_data.columns)} encontradas")
        
        # Check minimum rows
        if len(data) < rules['min_rows']:
            raise ValueError(f"{rules['message']} - {len(data)} observações encontradas")
        
        # Check for NaN/infinite values in numeric data
        if numeric_data.isnull().any().any():
            raise ValueError(f"Dados contêm valores ausentes (NaN)")
        
        if np.isinf(numeric_data.values).any():
            raise ValueError(f"Dados contêm valores infinitos")
        
        # Special validation for PCA
        if plot_type == 'PCA Biplot':
            if len(data) <= len(numeric_data.columns):
                raise ValueError("PCA requer mais observações do que variáveis")
        
        return True

    def _get_default_title(self, graph_type, use_rankings):
        """Generate intelligent default title based on graph type."""
        data_type = "Rankings" if use_rankings else "Índices"
        titles = {
            'Gráfico de Dispersão': f'Análise de Dispersão - {data_type}',
            'Gráfico 3D': f'Visualização 3D - {data_type}',
            'PCA Biplot': f'PCA Biplot - {data_type}',
            'Dendrograma': f'Análise de Clusters - {data_type}',
            'Gráfico de Barras': f'Comparação de Valores - {data_type}',
            'Histograma': f'Distribuição de Frequências - {data_type}',
            'Box Plot': f'Análise de Quartis - {data_type}',
            'Heatmap de Correlação': f'Mapa de Correlação - {data_type}'
        }
        return titles.get(graph_type, f'Análise Gráfica - {data_type}')

    def _get_default_xlabel(self, graph_type, data):
        """Generate intelligent default xlabel based on graph type and data."""
        if graph_type == 'Gráfico de Dispersão':
            try:
                x_var = self.scatter_x_combo.currentText() if hasattr(self, 'scatter_x_combo') else None
                return x_var if x_var else 'Variável X'
            except:
                return 'Variável X'
        elif graph_type == 'Gráfico 3D':
            try:
                x_var = self.graph_3d_x_combo.currentText() if hasattr(self, 'graph_3d_x_combo') else None
                return x_var if x_var else 'Variável X'
            except:
                return 'Variável X'
        elif graph_type == 'PCA Biplot':
            return 'Primeira Componente Principal'
        elif graph_type == 'Dendrograma':
            return 'Genótipos'
        elif graph_type == 'Gráfico de Barras':
            return 'Genótipos'
        elif graph_type == 'Histograma':
            return 'Valores'
        elif graph_type == 'Box Plot':
            return 'Categorias'
        elif graph_type == 'Heatmap de Correlação':
            return 'Variáveis'
        return 'Eixo X'

    def _get_default_ylabel(self, graph_type, data):
        """Generate intelligent default ylabel based on graph type and data."""
        if graph_type == 'Gráfico de Dispersão':
            try:
                y_var = self.scatter_y_combo.currentText() if hasattr(self, 'scatter_y_combo') else None
                return y_var if y_var else 'Variável Y'
            except:
                return 'Variável Y'
        elif graph_type == 'Gráfico 3D':
            try:
                y_var = self.graph_3d_y_combo.currentText() if hasattr(self, 'graph_3d_y_combo') else None
                return y_var if y_var else 'Ys'
            except:
                return 'Ys'
        elif graph_type == 'PCA Biplot':
            return 'Segunda Componente Principal'
        elif graph_type == 'Dendrograma':
            return 'Distância'
        elif graph_type == 'Gráfico de Barras':
            return 'Valores'
        elif graph_type == 'Histograma':
            return 'Frequência'
        elif graph_type == 'Box Plot':
            return 'Valores'
        elif graph_type == 'Heatmap de Correlação':
            return 'Variáveis'
        return 'Eixo Y'

    def _handle_auto_update(self):
        """Handle automatic graph update when controls change."""
        try:
            # Only update if we have data and auto-update is enabled
            if hasattr(self, 'auto_update_enabled') and self.auto_update_enabled:
                if not self.indices.empty:
                    # Add a small delay to avoid too many rapid updates
                    if hasattr(self, '_update_timer'):
                        self._update_timer.stop()
                    else:
                        from PyQt5.QtCore import QTimer
                        self._update_timer = QTimer()
                        self._update_timer.setSingleShot(True)
                        self._update_timer.timeout.connect(self._delayed_graph_update)
                    self._update_timer.start(500)  # 500ms delay
        except Exception as e:
            print(f"Warning: Auto-update failed: {e}")

    def _delayed_graph_update(self):
        """Delayed graph update for auto-preview."""
        try:
            self.generate_graph()
        except Exception as e:
            print(f"Warning: Delayed graph update failed: {e}")

    def toggle_auto_update(self, enabled):
        """Toggle auto-update mode."""
        self.auto_update_enabled = enabled
        if hasattr(self, 'auto_update_check'):
            self.auto_update_check.setChecked(enabled)

    def _on_scroll_zoom(self, event):
        """Handle mouse wheel zoom functionality."""
        try:
            if event.inaxes is None:
                return
            
            # Get current axis limits
            ax = event.inaxes
            xlims = ax.get_xlim()
            ylims = ax.get_ylim()
            
            # Calculate zoom factor
            base_scale = 1.5
            if event.button == 'up':
                # Zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                # Zoom out
                scale_factor = base_scale
            else:
                return
            
            # Get mouse position in data coordinates
            xdata = event.xdata
            ydata = event.ydata
            
            # Calculate new limits
            x_left = xdata - (xdata - xlims[0]) * scale_factor
            x_right = xdata + (xlims[1] - xdata) * scale_factor
            y_bottom = ydata - (ydata - ylims[0]) * scale_factor
            y_top = ydata + (ylims[1] - ydata) * scale_factor
            
            # Set new limits
            ax.set_xlim([x_left, x_right])
            ax.set_ylim([y_bottom, y_top])
            
            # Refresh the canvas
            event.canvas.draw()
            
        except Exception as e:
            print(f"Warning: Scroll zoom failed: {e}")

    def generate_graph(self):
        """Generates and displays a graph based on user selections."""
        if self.indices.empty:
            QMessageBox.warning(self, "Dados Ausentes", "Calcule os índices primeiro para gerar gráficos.")
            return

        # Clear previous graph safely
        try:
            if hasattr(self, 'current_graph_fig') and self.current_graph_fig: 
                plt.close(self.current_graph_fig)
                self.current_graph_fig = None
            if hasattr(self, 'current_canvas') and self.current_canvas:
                self.current_canvas.deleteLater()
                self.current_canvas = None
        except Exception as e:
            print(f"Warning: Could not close previous graph: {e}")
        
        # Clear widgets from layout safely
        try:
            while self.graph_preview_layout.count():
                child = self.graph_preview_layout.takeAt(0)
                if child:
                    if child.widget():
                        child.widget().setParent(None)
                        child.widget().deleteLater()
                    elif child.layout():
                        self._clear_layout(child.layout())
        except Exception as e:
            print(f"Warning: Could not clear layout: {e}")
            # If clearing fails, try to create a new layout
            try:
                self.graph_preview_layout = QVBoxLayout(self.graph_preview_widget)
            except Exception:
                pass

        # Get parameters from UI
        logging.info(f"Indices columns: {self.indices.columns}")
        logging.info(f"Rankings columns: {self.rankings_df.columns}")
        graph_type = self.graph_type_combo.currentText()
        logging.info(f"Generating graph: {graph_type}")
        logging.info(f"Scatter X: {self.scatter_x_combo.currentText()}, Scatter Y: {self.scatter_y_combo.currentText()}")
        use_rankings = self.use_rankings_graph.isChecked()
        
        # Check if rankings are available when requested
        if use_rankings and self.rankings_df.empty:
            QMessageBox.warning(self, "Rankings Ausentes", "Calcule os rankings primeiro para usar esta opção.")
            return
        
        # Choose the appropriate dataset
        data_to_use = self.rankings_df if use_rankings else self.indices
        
        # Validate data before plotting
        try:
            self._validate_data_for_plot(data_to_use, graph_type)
        except ValueError as e:
            QMessageBox.warning(self, "Dados Inválidos", f"Não é possível gerar o gráfico:\n{str(e)}")
            return
        except Exception as e:
            QMessageBox.warning(self, "Erro de Validação", f"Erro ao validar dados:\n{str(e)}")
            return
        
        # Get style options - use defaults if widgets don't exist
        try:
            hue_by = self.hue_combo.currentText()
        except AttributeError:
            hue_by = 'Nenhum'
        if hue_by == 'Nenhum':
            hue_by = None
        elif hue_by == 'Classificação por RP':
            # Use a coluna de classificação por RP dos índices
            if hasattr(self, 'indices') and 'Classificação por RP' in self.indices.columns:
                hue_by = 'Classificação por RP'
                # Para gráficos que usam rankings, precisamos adicionar a classificação temporariamente
                if use_rankings:
                    classification_map = dict(zip(self.indices['Genotipo'], self.indices['Classificação por RP']))
                    data_to_use = self.rankings_df.copy()
                    data_to_use['Classificação por RP'] = data_to_use['Genotipo'].map(classification_map)
            else:
                QMessageBox.warning(self, "Coluna Ausente", "A coluna 'Classificação por RP' não está disponível. Desabilitando agrupamento de cores.")
                hue_by = None
        elif hue_by == 'Classificação por MFV':
            # Use a coluna de classificação por MFV
            print(f"\n=== DEBUG MFV CLASSIFICATION ===")
            print(f"hasattr(self, 'mfv_df'): {hasattr(self, 'mfv_df')}")
            if hasattr(self, 'mfv_df'):
                print(f"self.mfv_df is not None: {self.mfv_df is not None}")
                if self.mfv_df is not None:
                    print(f"MFV columns: {list(self.mfv_df.columns)}")
                    print(f"'Classe_Tolerância' in columns: {'Classe_Tolerância' in self.mfv_df.columns}")
                    print(f"MFV shape: {self.mfv_df.shape}")
            
            if hasattr(self, 'mfv_df') and self.mfv_df is not None and 'Classe_Tolerância' in self.mfv_df.columns:
                hue_by = 'Classe_Tolerância'
                print(f"Using MFV classification with column: {hue_by}")
                
                # Adicionar a classificação MFV aos dados
                mfv_classification_map = dict(zip(self.mfv_df['Genotipo'], self.mfv_df['Classe_Tolerância']))
                print(f"MFV classification map (first 5): {dict(list(mfv_classification_map.items())[:5])}")
                
                if use_rankings:
                    data_to_use = self.rankings_df.copy()
                    data_to_use['Classe_Tolerância'] = data_to_use['Genotipo'].map(mfv_classification_map)
                    print(f"Added to rankings - unique classes: {data_to_use['Classe_Tolerância'].unique()}")
                else:
                    data_to_use = self.indices.copy()
                    data_to_use['Classe_Tolerância'] = data_to_use['Genotipo'].map(mfv_classification_map)
                    print(f"Added to indices - unique classes: {data_to_use['Classe_Tolerância'].unique()}")
            else:
                print("MFV analysis not available - showing warning")
                QMessageBox.warning(self, "Análise MFV Não Disponível", "Execute primeiro a análise MFV para usar esta classificação.")
                hue_by = None
        # Cluster option removed as requested
        
        # Choose the appropriate dataset (only if not already set by MFV processing)
        if 'data_to_use' not in locals():
            data_to_use = self.rankings_df if use_rankings else self.indices
        
        if hue_by and hue_by not in data_to_use.columns:
            QMessageBox.warning(self, "Coluna Ausente", f"A coluna '{hue_by}' não está disponível. Desabilitando agrupamento de cores.")
            hue_by = None

        # Get style options - use defaults if widgets don't exist
        try:
            palette = self.palette_combo.currentText()
            show_labels = self.show_labels_check.isChecked()
            title = self.title_input.text().strip()
            xlabel = self.xlabel_input.text().strip()
            ylabel = self.ylabel_input.text().strip()
        except AttributeError:
            palette = 'viridis'
            show_labels = False
            title = ""
            xlabel = ""
            ylabel = ""

        # Generate intelligent default labels if empty
        if not title:
            title = self._get_default_title(graph_type, use_rankings)
        if not xlabel:
            xlabel = self._get_default_xlabel(graph_type, data_to_use)
        if not ylabel:
            ylabel = self._get_default_ylabel(graph_type, data_to_use)

        # Get font settings - use defaults if widgets don't exist
        try:
            font_settings = {
                'title': self.title_font_combo.currentText(),
                'axes': self.axes_font_combo.currentText(),
                'tick': self.tick_font_combo.currentText(),
                'legend': self.legend_font_combo.currentText(),
                'annotation': self.annotation_font_combo.currentText(),
                'general': self.general_font_combo.currentText()
            }
        except AttributeError:
            font_settings = {
                'title': 'Arial',
                'axes': 'Arial',
                'tick': 'Arial',
                'legend': 'Arial',
                'annotation': 'Arial',
                'general': 'Arial'
            }

        # Criar figura 3D se necessário
        if graph_type == 'Gráfico 3D':
            fig = plt.figure(figsize=(12, 10))  # Aumentar tamanho para 3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Apply general font
        plt.rcParams['font.family'] = font_settings['general']
        
        try:
            if graph_type == 'Gráfico de Dispersão':
                self._generate_scatter_plot(ax, data_to_use, hue_by, palette, show_labels, font_settings)
            elif graph_type == 'Gráfico 3D':
                self._generate_3d_plot(ax, data_to_use, hue_by, palette, show_labels, font_settings)
            elif graph_type == 'PCA Biplot':
                self._generate_pca_plot(ax, data_to_use, hue_by, palette, show_labels, font_settings)
                # Integração: atualizar automaticamente a aba de Resultados da PCA
                self._update_pca_results_from_graph(data_to_use, use_rankings)
            elif graph_type == 'Dendrograma':
                self._generate_dendrogram(ax, data_to_use, font_settings)
            elif graph_type == 'Gráfico de Barras':
                self._generate_bar_plot(ax, data_to_use, hue_by, palette, show_labels, font_settings)
            elif graph_type == 'Histograma':
                self._generate_histogram(ax, data_to_use, hue_by, palette, show_labels, font_settings)
            elif graph_type == 'Box Plot':
                self._generate_boxplot(ax, data_to_use, hue_by, palette, show_labels, font_settings)
            elif graph_type == 'Heatmap de Correlação':
                self._generate_heatmap(ax, data_to_use, font_settings)
            
            # Apply custom labels and title
            if title:
                ax.set_title(title, fontfamily=font_settings['title'], fontsize=16, fontweight='bold')
            
            # For PCA, don't override labels if they're already set by PCA method
            current_xlabel = ax.get_xlabel()
            current_ylabel = ax.get_ylabel()
            
            # Check if labels are from PCA (either with variance or just PC format)
            pca_xlabel_set = (graph_type == 'PCA Biplot' and 
                             (current_xlabel.startswith('PC') or 'da variância' in current_xlabel))
            pca_ylabel_set = (graph_type == 'PCA Biplot' and 
                             (current_ylabel.startswith('PC') or 'da variância' in current_ylabel))
            
            if xlabel and not pca_xlabel_set:
                ax.set_xlabel(xlabel, fontfamily=font_settings['axes'], fontsize=12)
            if ylabel and not pca_ylabel_set:
                ax.set_ylabel(ylabel, fontfamily=font_settings['axes'], fontsize=12)
            
            # Handle Z-label for 3D plots
            if graph_type == 'Gráfico 3D' and hasattr(ax, 'set_zlabel'):
                try:
                    zlabel = self.zlabel_input.text().strip() if hasattr(self, 'zlabel_input') else None
                    if zlabel:
                        ax.set_zlabel(zlabel, fontfamily=font_settings['axes'], fontsize=12)
                except:
                    pass  # If zlabel fails, continue without it
            
            # Apply font to tick labels
            ax.tick_params(axis='both', which='major', labelsize=10)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily(font_settings['tick'])
            
            # De-duplicate legend
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())

            # Adjust layout for better space utilization
            if graph_type == 'Gráfico 3D':
                # 3D plots need more space for axes labels
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            else:
                plt.tight_layout()
            
            # Clear previous widgets from layout
            for i in reversed(range(self.graph_preview_layout.count())):
                child = self.graph_preview_layout.takeAt(i)
                if child.widget():
                    child.widget().deleteLater()
            
            # Display graph with navigation toolbar
            canvas = FigureCanvas(fig)
            canvas.setMinimumSize(500, 400)  # Tamanho mínimo maior para melhor visualização
            canvas.setSizePolicy(canvas.sizePolicy().Expanding, canvas.sizePolicy().Expanding)
            
            # Add navigation toolbar for zoom/pan functionality
            if NAVIGATION_AVAILABLE:
                try:
                    toolbar = NavigationToolbar(canvas, self)
                    toolbar.setStyleSheet("QToolBar { background-color: #f0f0f0; }")
                    
                    # Customize toolbar - remove some buttons for cleaner interface
                    actions = toolbar.actions()
                    for action in actions:
                        if action.text() in ['Configure subplots', 'Customize']:
                            action.setVisible(False)
                    
                    # Add toolbar and canvas
                    self.graph_preview_layout.addWidget(toolbar)
                    self.graph_preview_layout.addWidget(canvas)
                    
                except Exception as e:
                    print(f"Warning: Could not create navigation toolbar: {e}")
                    self.graph_preview_layout.addWidget(canvas)
            else:
                self.graph_preview_layout.addWidget(canvas)
            
            # Enable interactive features
            canvas.mpl_connect('button_press_event', self.on_graph_click)
            
            # Add mouse wheel zoom functionality
            canvas.mpl_connect('scroll_event', self._on_scroll_zoom)
            
            # Force multiple refresh attempts
            canvas.draw()  # Force drawing of the canvas
            canvas.flush_events()  # Flush any pending GUI events
            plt.draw()  # Force matplotlib drawing
            canvas.show()  # Ensure canvas is visible
            
            self.current_graph_fig = fig
            self.current_canvas = canvas
            
            self.status_bar.showMessage("Gráfico gerado com sucesso!", 3000)
            self.update_ui_state()  # Update button states
            
        except Exception as e:
            QMessageBox.critical(self, "Erro ao Gerar Gráfico", 
                               f"Erro ao gerar gráfico:\n{str(e)}")
            logging.error(f"Graph generation error: {e}", exc_info=True)
            plt.close(fig)

    def _generate_scatter_plot(self, ax, data, hue_by, palette, show_labels, font_settings):
        """Generate scatter plot with user-selectable axes and improved error handling."""
        try:
            # Get selected variables from comboboxes
            x_var = self.scatter_x_combo.currentText() if hasattr(self, 'scatter_x_combo') else None
            y_var = self.scatter_y_combo.currentText() if hasattr(self, 'scatter_y_combo') else None

            logging.info(f"Scatter plot - x_var: '{x_var}', y_var: '{y_var}', data shape: {data.shape}")
            
            # Validate variable selection
            if not x_var or not y_var:
                raise ValueError("Selecione as variáveis para os eixos X e Y.")
            
            if x_var not in data.columns:
                raise ValueError(f"Variável X '{x_var}' não encontrada. Disponíveis: {list(data.columns)}")
            
            if y_var not in data.columns:
                raise ValueError(f"Variável Y '{y_var}' não encontrada. Disponíveis: {list(data.columns)}")
            
            # Check for numeric data
            if not pd.api.types.is_numeric_dtype(data[x_var]):
                raise ValueError(f"Variável X '{x_var}' deve ser numérica")
            
            if not pd.api.types.is_numeric_dtype(data[y_var]):
                raise ValueError(f"Variável Y '{y_var}' deve ser numérica")

        except Exception as e:
            raise ValueError(f"Erro na configuração do gráfico de dispersão: {str(e)}")

        # Debug data before plotting
        print(f"DEBUG: data[{x_var}] sample: {data[x_var].head()}")
        print(f"DEBUG: data[{y_var}] sample: {data[y_var].head()}")
        print(f"DEBUG: data[{x_var}] null count: {data[x_var].isnull().sum()}")
        print(f"DEBUG: data[{y_var}] null count: {data[y_var].isnull().sum()}")
        print(f"DEBUG: data[{x_var}] min/max: {data[x_var].min()} / {data[x_var].max()}")
        print(f"DEBUG: data[{y_var}] min/max: {data[y_var].min()} / {data[y_var].max()}")

        # Remove any NaN values for plotting
        plot_data = data[[x_var, y_var]].dropna()
        print(f"DEBUG: Original data rows: {len(data)}, After dropna: {len(plot_data)}")
        
        if len(plot_data) == 0:
            raise ValueError("Não há dados válidos para plotar após remover valores nulos.")

        # Create the scatter plot
        try:
            if hue_by and hue_by in data.columns:
                # Ordenar classes MFV na ordem desejada para manter consistência nas cores
                if hue_by == 'Classe_Tolerância':
                    mfv_order = ['Altamente Tolerante', 'Tolerante', 'Moderadamente Tolerante', 'Sensível', 'Altamente Sensível']
                    data[hue_by] = pd.Categorical(data[hue_by], categories=mfv_order, ordered=True)
                
                sns.scatterplot(data=data, x=x_var, y=y_var, hue=hue_by, palette=palette, s=100, alpha=0.7, ax=ax)
            else:
                # Use matplotlib directly for better reliability
                ax.scatter(plot_data[x_var], plot_data[y_var], s=100, alpha=0.7, c='steelblue')
            
            # Force axis limits to ensure points are visible
            if len(plot_data) > 0:
                x_min, x_max = float(plot_data[x_var].min()), float(plot_data[x_var].max())
                y_min, y_max = float(plot_data[y_var].min()), float(plot_data[y_var].max())
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                # Handle case where all values are the same
                x_margin = max(x_range * 0.1, 1) if x_range > 0 else 1
                y_margin = max(y_range * 0.1, 1) if y_range > 0 else 1
                
                ax.set_xlim(x_min - x_margin, x_max + x_margin)
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
                
        except Exception as e:
            raise ValueError(f"Erro ao criar gráfico de dispersão: {e}")
        
        # Add labels to points if requested
        if show_labels and 'Genotipo' in data.columns:
            for idx, row in data.iterrows():
                ax.annotate(row['Genotipo'], 
                          (row[x_var], row[y_var]), 
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.8, fontfamily=font_settings['annotation'])
        
        # Set axis labels and title (can be overridden by user input later)
        xlabel = ax.set_xlabel(x_var, fontfamily=font_settings['axes'])
        ylabel = ax.set_ylabel(y_var, fontfamily=font_settings['axes'])
        title = ax.set_title(f'{x_var} vs {y_var}', fontfamily=font_settings['title'])
        xlabel.set_picker(True)
        ylabel.set_picker(True)
        title.set_picker(True)
        
        # Add grid for better visualization
        ax.grid(True, alpha=0.3)

    def _generate_3d_plot(self, ax, data, hue_by, palette, show_labels, font_settings):
        """Generate interactive 3D scatter plot with user-selectable axes."""
        try:
            # Get selected variables from 3D comboboxes
            x_var = self.graph_3d_x_combo.currentText() if hasattr(self, 'graph_3d_x_combo') else None
            y_var = self.graph_3d_y_combo.currentText() if hasattr(self, 'graph_3d_y_combo') else None
            z_var = self.graph_3d_z_combo.currentText() if hasattr(self, 'graph_3d_z_combo') else None

            logging.info(f"3D plot - x_var: '{x_var}', y_var: '{y_var}', z_var: '{z_var}', data shape: {data.shape}")
            
            # Validate variable selection
            if not x_var or not y_var or not z_var:
                raise ValueError("Selecione as variáveis para os eixos X, Y e Z.")
            
            if x_var not in data.columns:
                raise ValueError(f"Variável X '{x_var}' não encontrada. Disponíveis: {list(data.columns)}")
            
            if y_var not in data.columns:
                raise ValueError(f"Variável Y '{y_var}' não encontrada. Disponíveis: {list(data.columns)}")
            
            if z_var not in data.columns:
                raise ValueError(f"Variável Z '{z_var}' não encontrada. Disponíveis: {list(data.columns)}")
            
            # Check for numeric data
            if not pd.api.types.is_numeric_dtype(data[x_var]):
                raise ValueError(f"Variável X '{x_var}' deve ser numérica")
            
            if not pd.api.types.is_numeric_dtype(data[y_var]):
                raise ValueError(f"Variável Y '{y_var}' deve ser numérica")
            
            if not pd.api.types.is_numeric_dtype(data[z_var]):
                raise ValueError(f"Variável Z '{z_var}' deve ser numérica")

        except Exception as e:
            raise ValueError(f"Erro na configuração do gráfico 3D: {str(e)}")

        # Preparar dados incluindo classificação e genótipo se necessário
        plot_columns = [x_var, y_var, z_var]
        if hue_by and hue_by in data.columns:
            plot_columns.append(hue_by)
        # Always include Genotipo for labels if available
        if 'Genotipo' in data.columns and 'Genotipo' not in plot_columns:
            plot_columns.append('Genotipo')
        
        plot_data = data[plot_columns].dropna()
        
        if len(plot_data) == 0:
            raise ValueError("Não há dados válidos para plotar após remover valores nulos.")

        # Debug info
        print(f"DEBUG 3D: x_var='{x_var}', y_var='{y_var}', z_var='{z_var}', hue_by='{hue_by}'")
        print(f"DEBUG 3D: plot_data shape: {plot_data.shape}")
        print(f"DEBUG 3D: Available columns: {list(data.columns)}")

        # Create the 3D scatter plot
        try:
            if hue_by and hue_by in plot_data.columns:
                print(f"DEBUG 3D: Using classification: {hue_by}")
                print(f"DEBUG 3D: Unique values in {hue_by}: {plot_data[hue_by].unique()}")
                
                # Correct MFV order mapping
                if 'MFV' in hue_by or 'Classe_Tolerância' in hue_by:
                    # Order from most tolerant to most sensitive
                    mfv_order = ['Altamente Tolerante', 'Tolerante', 'Moderadamente Tolerante', 'Sensível', 'Altamente Sensível']
                    # Define colors for MFV classes (blue to red)
                    mfv_colors = ['#0000FF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000']  # Blue, Green, Yellow, Orange, Red
                    color_map = dict(zip(mfv_order, mfv_colors))
                    
                    # Get unique categories in order
                    available_categories = [cat for cat in mfv_order if cat in plot_data[hue_by].unique()]
                    
                    for category in available_categories:
                        mask = plot_data[hue_by] == category
                        category_data = plot_data[mask]
                        if len(category_data) > 0:
                            ax.scatter(category_data[x_var].values, 
                                     category_data[y_var].values, 
                                     category_data[z_var].values,
                                     c=color_map[category], 
                                     label=category, 
                                     alpha=0.8,
                                     s=80)
                else:
                    # Regular classification (RP)
                    categories = plot_data[hue_by].unique()
                    colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))
                    
                    for i, category in enumerate(categories):
                        mask = plot_data[hue_by] == category
                        category_data = plot_data[mask]
                        if len(category_data) > 0:
                            ax.scatter(category_data[x_var].values, 
                                     category_data[y_var].values, 
                                     category_data[z_var].values,
                                     c=[colors[i]], 
                                     label=str(category), 
                                     alpha=0.8,
                                     s=80)
                
                # Add legend
                ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(0.02, 0.98))
                
            else:
                print("DEBUG 3D: No classification, single color plot")
                # Single color plot
                ax.scatter(plot_data[x_var].values, 
                         plot_data[y_var].values, 
                         plot_data[z_var].values,
                         alpha=0.8, 
                         s=80,
                         c='steelblue')
                
        except Exception as e:
            print(f"DEBUG 3D: Error in scatter plot creation: {e}")
            raise ValueError(f"Erro ao criar gráfico 3D: {e}")
        
        # Add labels to points if requested
        print(f"DEBUG 3D: show_labels={show_labels}, columns in plot_data: {list(plot_data.columns)}")
        if show_labels and 'Genotipo' in plot_data.columns:
            print(f"DEBUG 3D: Adding labels for {len(plot_data)} genotypes")
            for idx, row in plot_data.iterrows():
                try:
                    ax.text(row[x_var], row[y_var], row[z_var], 
                           str(row['Genotipo']), 
                           fontsize=9, alpha=0.9,
                           fontweight='bold')
                except Exception as e:
                    print(f"DEBUG 3D: Error adding label for row {idx}: {e}")
        elif show_labels:
            print("DEBUG 3D: Labels requested but Genotipo column not available")
        else:
            print("DEBUG 3D: Labels not requested")
        
        # Set axis labels correctly
        ax.set_xlabel(x_var, fontfamily=font_settings['axes'], fontsize=12, labelpad=10)
        ax.set_ylabel(y_var, fontfamily=font_settings['axes'], fontsize=12, labelpad=10)
        ax.set_zlabel(z_var, fontfamily=font_settings['axes'], fontsize=12, labelpad=10)
        
        # Set title with proper formatting
        ax.set_title(f'Visualização 3D: {x_var} × {y_var} × {z_var}', 
                    fontfamily=font_settings['title'], fontsize=14, fontweight='bold', pad=20)
        
        # Improve 3D visualization
        ax.view_init(elev=20, azim=45)  # Better default viewing angle
        ax.grid(True, alpha=0.3)
        
        # Enable interactivity - mouse rotation and zoom
        ax.mouse_init()

    def _generate_bar_plot(self, ax, data, hue_by, palette, show_labels, font_settings):
        """Generate bar plot with multiple variable support."""
        # Get variables from the multi-selection list
        vars_to_plot = [self.multi_var_list_selected.item(i).text() for i in range(self.multi_var_list_selected.count())]
        
        if not vars_to_plot:
            return

        # Ensure Genotipo column exists for melting
        id_col = 'Genotipo'
        if id_col not in data.columns:
             raise ValueError("Coluna 'Genotipo' não encontrada nos dados.")

        # Melt the dataframe to prepare for grouped bar plot
        plot_data = data.melt(id_vars=[id_col], value_vars=vars_to_plot, 
                              var_name='Índice', value_name='Valor')
        
        # Create the plot
        sns.barplot(data=plot_data, x=id_col, y='Valor', hue='Índice', palette=palette, ax=ax)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
        
        ax.set_xlabel('Genótipos', fontfamily=font_settings['axes'])
        ax.set_ylabel('Valor', fontfamily=font_settings['axes'])
        ax.set_title('Gráfico de Barras Comparativo', fontfamily=font_settings['title'])

    def _generate_histogram(self, ax, data, hue_by, palette, show_labels, font_settings):
        """Generate histogram with multiple variable support."""
        # Get variables from the multi-selection list
        vars_to_plot = [self.multi_var_list_selected.item(i).text() for i in range(self.multi_var_list_selected.count())]
        
        if not vars_to_plot:
            return

        # Plot each selected variable
        for var in vars_to_plot:
            sns.histplot(data[var], kde=True, ax=ax, label=var, alpha=0.6, multiple="layer")

        ax.set_xlabel('Valor', fontfamily=font_settings['axes'])
        ax.set_ylabel('Frequência', fontfamily=font_settings['axes'])
        ax.set_title('Histograma Comparativo', fontfamily=font_settings['title'])
        ax.legend()

    def _generate_boxplot(self, ax, data, hue_by, palette, show_labels, font_settings):
        """Generate box plot with multiple variable support."""
        # Get variables from the multi-selection list
        vars_to_plot = [self.multi_var_list_selected.item(i).text() for i in range(self.multi_var_list_selected.count())]
        
        if not vars_to_plot:
            return

        # Melt the dataframe to prepare for box plot
        plot_data = data.melt(value_vars=vars_to_plot, 
                              var_name='Índice', value_name='Valor')
        
        # Create the plot
        sns.boxplot(data=plot_data, x='Índice', y='Valor', palette=palette, ax=ax)
        
        ax.set_xlabel('Índice', fontfamily=font_settings['axes'])
        ax.set_ylabel('Valor', fontfamily=font_settings['axes'])
        ax.set_title('Box Plot Comparativo', fontfamily=font_settings['title'])

    def _generate_heatmap(self, ax, data, font_settings):
        """Generate correlation heatmap with improved options."""
        # Get heatmap options - use default values if widgets don't exist
        try:
            corr_method = self.heatmap_method_combo.currentText()
            show_values = self.heatmap_values_check.isChecked()
        except AttributeError:
            corr_method = 'pearson'  # Default correlation method
            show_values = True       # Default to show values
        
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=np.number)
        if numeric_data.empty:
            raise ValueError("Nenhuma coluna numérica encontrada para correlação")
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr(method=corr_method)
        
        # Clean column/index names if using rankings
        if 'Rank_' in str(corr_matrix.columns[0]):
            corr_matrix.columns = [col.replace('Rank_', '') for col in corr_matrix.columns]
            corr_matrix.index = [idx.replace('Rank_', '') for idx in corr_matrix.index]
        
        # Generate heatmap
        # Get palette for heatmap
        try:
            cmap_value = self.palette_combo.currentText()
        except AttributeError:
            cmap_value = 'viridis'
            
        sns.heatmap(corr_matrix, 
                   annot=show_values, 
                   cmap=cmap_value,
                   center=0, 
                   square=True,
                   fmt='.2f' if show_values else '',
                   cbar_kws={"shrink": .8},
                   ax=ax)
        
        # Apply custom title if provided, otherwise use default
        try:
            title = self.title_input.text()
        except AttributeError:
            title = ""
        if title:
            ax.set_title(title, fontfamily=font_settings['title'], fontsize=14)
        else:
            data_type = " (Rankings)" if 'Rank_' in str(data.columns[0]) else " (Índices)"
            ax.set_title(f'Heatmap de Correlação{data_type}', 
                        fontfamily=font_settings['title'], fontsize=14)
        
        # Rotate labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', 
                          fontfamily=font_settings['tick'])
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, 
                          fontfamily=font_settings['tick'])

    def _generate_pca_plot(self, ax, data, hue_by, palette, show_labels, font_settings):
        """Generate advanced PCA biplot with customizable components, matrix type, and variance display."""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ValueError("Scikit-learn é necessário para PCA. Execute: pip install scikit-learn")
        
        # Get PCA options - use default values if widgets don't exist
        try:
            n_components = self.pca_components_spin.value()
            pc_x = self.pca_component_x_combo.currentText()  # e.g., "PC1"
            pc_y = self.pca_component_y_combo.currentText()  # e.g., "PC2"
            matrix_type = self.pca_matrix_combo.currentText()  # "Correlação" ou "Covariância"
            show_variance = self.pca_show_variance_check.isChecked()
        except AttributeError:
            n_components = 2
            pc_x, pc_y = "PC1", "PC2"
            matrix_type = "Correlação"
            show_variance = True
        
        # DEBUG: Print PCA settings
        print(f"\n=== PCA DEBUG ===")
        print(f"N° Componentes solicitados: {n_components}")
        print(f"Componente X: {pc_x}")
        print(f"Componente Y: {pc_y}")
        print(f"Tipo de matriz: {matrix_type}")
        print(f"Mostrar variância: {show_variance}")
        print(f"Data shape: {data.shape}")
        
        # Extract component indices (PC1 -> 0, PC2 -> 1, etc.)
        try:
            pc_x_idx = int(pc_x.replace('PC', '')) - 1
            pc_y_idx = int(pc_y.replace('PC', '')) - 1
        except (ValueError, AttributeError):
            pc_x_idx, pc_y_idx = 0, 1  # Default to PC1, PC2
        
        print(f"Índices calculados: pc_x_idx={pc_x_idx}, pc_y_idx={pc_y_idx}")
        
        # Validate component selection
        max_components = min(n_components, 10)
        print(f"Máximo de componentes: {max_components}")
        if pc_x_idx >= max_components or pc_y_idx >= max_components:
            raise ValueError(f"Componente selecionado excede o número máximo ({max_components})")
        
        # Select numeric columns and validate data
        numeric_data = data.select_dtypes(include=np.number)
        if numeric_data.empty:
            raise ValueError("Nenhuma coluna numérica encontrada para PCA")
        
        if len(numeric_data.columns) < 2:
            raise ValueError("PCA requer pelo menos 2 variáveis numéricas")

        # Prepare data based on matrix type
        X = numeric_data.values
        
        if matrix_type == "Correlação":
            # Para matriz de correlação, sempre padronizar
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        # Para matriz de covariância, usar dados brutos (sem padronização)
        
        # Perform PCA with sufficient components
        required_components = max(pc_x_idx + 1, pc_y_idx + 1, n_components)
        actual_components = min(required_components, len(numeric_data.columns))
        print(f"Componentes necessários: {required_components}")
        print(f"Componentes que serão calculados: {actual_components}")
        print(f"Variáveis numéricas: {len(numeric_data.columns)} - {list(numeric_data.columns)}")
        
        pca = PCA(n_components=actual_components)
        X_pca = pca.fit_transform(X)
        
        print(f"PCA shape resultante: {X_pca.shape}")
        print(f"Variância explicada por componente: {pca.explained_variance_ratio_}")
        print(f"Variância explicada total: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Obter nomes dos genótipos para as coordenadas
        genotype_names = []
        possible_genotype_cols = ['Genótipo', 'Genotype', 'ID', 'Nome', 'Name', 'Genotipo']
        genotype_col_found = None
        
        for col in possible_genotype_cols:
            if col in data.columns:
                genotype_col_found = col
                genotype_names = data[col].tolist()
                break
        
        if not genotype_names:
            # Fallback: usar índices ou criar nomes genéricos
            if hasattr(data, 'index') and not data.index.equals(pd.RangeIndex(len(data))):
                genotype_names = data.index.tolist()
            else:
                genotype_names = [f'Genótipo_{i+1}' for i in range(len(data))]
        
        print(f"DEBUG: Usando nomes de genótipos: {genotype_names[:5]}... (primeiros 5)")
        print(f"DEBUG: Coluna de genótipos encontrada: {genotype_col_found}")
        print(f"DEBUG: Total de genótipos: {len(genotype_names)}")
        
        # Armazenar resultados do PCA para relatórios
        coordinates_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(actual_components)])
        
        # Garantir que temos o número correto de nomes para as coordenadas
        print(f"DEBUG: Comprimento dos arrays: genotype_names={len(genotype_names)}, coordinates_df={len(coordinates_df)}")
        
        if len(genotype_names) == len(coordinates_df):
            coordinates_df.index = genotype_names
            print(f"DEBUG: Nomes atribuídos diretamente - primeiro: {coordinates_df.index[0]}")
        elif len(genotype_names) > len(coordinates_df):
            coordinates_df.index = genotype_names[:len(coordinates_df)]
            print(f"DEBUG: Nomes truncados - primeiro: {coordinates_df.index[0]}")
        else:
            # Se temos menos nomes que coordenadas, completar com nomes genéricos
            extended_names = genotype_names + [f'Genótipo_{i+1}' for i in range(len(genotype_names), len(coordinates_df))]
            coordinates_df.index = extended_names
            print(f"DEBUG: Nomes estendidos - primeiro: {coordinates_df.index[0]}")
        
        print(f"DEBUG: Índice final do coordinates_df: {coordinates_df.index.tolist()[:5]}... (primeiros 5)")
        
        # Verificar se o índice está sendo preservado corretamente
        print(f"DEBUG: Tipo do índice: {type(coordinates_df.index)}")
        print(f"DEBUG: Valores únicos no índice: {coordinates_df.index.nunique()}")
        print(f"DEBUG: Exemplo de valor do índice: '{coordinates_df.index[0]}' (tipo: {type(coordinates_df.index[0])})")
        
        self.current_pca_results = {
            'pca_object': pca,
            'components': X_pca,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'explained_variance': pca.explained_variance_,
            'eigenvalues': pca.explained_variance_,  # Autovalores
            'eigenvectors': pca.components_,  # Autovetores (componentes principais)
            'feature_names': list(numeric_data.columns),
            'n_components': actual_components,
            'matrix_type': matrix_type,
            'loadings': pca.components_.T * np.sqrt(pca.explained_variance_),
            'coordinates': coordinates_df,  # DataFrame com índice preservado
            'original_data': numeric_data,
            'genotype_names': genotype_names
        }
        print(f"DEBUG: Resultados PCA armazenados com {actual_components} componentes")
        
        # Validate that we have enough components
        if X_pca.shape[1] <= max(pc_x_idx, pc_y_idx):
            raise ValueError(f"Não há componentes suficientes. Máximo disponível: PC{X_pca.shape[1]}")
        
        # Get the selected components for plotting
        x_values = X_pca[:, pc_x_idx]
        y_values = X_pca[:, pc_y_idx]
        print(f"Valores X (PC{pc_x_idx+1}): min={x_values.min():.3f}, max={x_values.max():.3f}")
        print(f"Valores Y (PC{pc_y_idx+1}): min={y_values.min():.3f}, max={y_values.max():.3f}")
        
        # Create scatter plot with color grouping
        if hue_by and hue_by in data.columns:
            unique_vals = data[hue_by].unique()
            
            # Ordenar classes MFV na ordem desejada (Altamente Tolerante -> Altamente Sensível)
            if hue_by == 'Classe_Tolerância':
                mfv_order = ['Altamente Tolerante', 'Tolerante', 'Moderadamente Tolerante', 'Sensível', 'Altamente Sensível']
                unique_vals = [val for val in mfv_order if val in unique_vals] + [val for val in unique_vals if val not in mfv_order]
            
            colors = sns.color_palette(palette, n_colors=len(unique_vals))
            color_dict = {val: colors[i] for i, val in enumerate(unique_vals)}
            for val in unique_vals:
                idx = data[hue_by] == val
                ax.scatter(x_values[idx], y_values[idx], alpha=0.7, s=100, 
                          label=str(val), color=color_dict[val])
            ax.legend(title=hue_by, prop={'family': font_settings.get('legend', 'DejaVu Sans')})
        else:
            ax.scatter(x_values, y_values, alpha=0.7, s=100)
        
        # Add sample labels if requested
        if show_labels:
            genotype_col = 'Genotipo' if 'Genotipo' in data.columns else 'Genótipo'
            if genotype_col in data.columns:
                for i, txt in enumerate(data[genotype_col]):
                    # Reduzir sobreposição usando texto menor e offset maior
                    ax.annotate(str(txt)[:8], (x_values[i], y_values[i]), 
                               xytext=(8, 8), textcoords='offset points',
                               fontsize=8, alpha=0.8, 
                               fontfamily=font_settings.get('annotation', 'DejaVu Sans'))
        
        # Add loading vectors (biplot) for selected components
        try:
            print(f"\n--- LOADING VECTORS DEBUG ---")
            print(f"PCA components shape: {pca.components_.shape}")
            print(f"Explained variance: {pca.explained_variance_}")
            
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            print(f"Loadings shape: {loadings.shape}")
            print(f"Usando componentes: X=PC{pc_x_idx+1} (idx={pc_x_idx}), Y=PC{pc_y_idx+1} (idx={pc_y_idx})")
            
            # Calcular ranges dos PCs selecionados
            x_range = abs(x_values.max() - x_values.min())
            y_range = abs(y_values.max() - y_values.min())
            max_pc_range = max(x_range, y_range)
            
            print(f"X range: {x_range:.3f}, Y range: {y_range:.3f}")
            print(f"Range máximo dos PCs: {max_pc_range:.3f}")
            
            # Calcular escala dos loadings baseada apenas nos componentes selecionados
            selected_loadings_x = loadings[:, pc_x_idx]
            selected_loadings_y = loadings[:, pc_y_idx]
            
            max_loading_magnitude = max(
                abs(selected_loadings_x).max(),
                abs(selected_loadings_y).max()
            )
            
            print(f"Magnitude máxima dos loadings: {max_loading_magnitude:.3f}")
            
            # Escala inteligente: loading vectors devem ocupar ~30% do range do PCA
            if max_loading_magnitude > 0:
                target_size = max_pc_range * 0.3
                scale_factor = target_size / max_loading_magnitude
            else:
                scale_factor = 1.0
            
            print(f"Fator de escala inteligente: {scale_factor:.3f}")
            
            for i, feature in enumerate(numeric_data.columns):
                loading_x = selected_loadings_x[i] * scale_factor
                loading_y = selected_loadings_y[i] * scale_factor
                
                print(f"  {feature}: loading_x={loading_x:.3f}, loading_y={loading_y:.3f}")
                
                # Tamanho da seta proporcional ao range (muito mais sutil)
                head_width = max_pc_range * 0.002   # Muito menor
                head_length = max_pc_range * 0.004  # Muito menor
                
                ax.arrow(0, 0, loading_x, loading_y, 
                        head_width=head_width, head_length=head_length, 
                        fc='navy', ec='navy', alpha=0.5, linewidth=0.5)  # Cor mais escura e linha mais fina
                
                # Posicionar texto das variáveis para reduzir sobreposição
                text_x = loading_x * 1.2
                text_y = loading_y * 1.2
                
                # Ajustar posição baseado no quadrante
                ha_setting = 'left' if text_x > 0 else 'right'
                va_setting = 'bottom' if text_y > 0 else 'top'
                
                ax.text(text_x, text_y, feature[:12], 
                       fontsize=9, ha=ha_setting, va=va_setting, 
                       fontfamily=font_settings.get('annotation', 'DejaVu Sans'),
                       fontweight='bold', alpha=0.9)
        except Exception as e:
            print(f"Warning: Could not add loading vectors: {e}")
            import traceback
            traceback.print_exc()
        
        # Set labels with variance explained
        var_explained = pca.explained_variance_ratio_
        
        print(f"\n--- LABELS DEBUG ---")
        print(f"show_variance: {show_variance}")
        print(f"var_explained length: {len(var_explained)}")
        print(f"pc_x_idx: {pc_x_idx}, pc_y_idx: {pc_y_idx}")
        print(f"Condições: pc_x_idx < len: {pc_x_idx < len(var_explained)}, pc_y_idx < len: {pc_y_idx < len(var_explained)}")
        
        if show_variance and pc_x_idx < len(var_explained) and pc_y_idx < len(var_explained):
            x_variance = var_explained[pc_x_idx]
            y_variance = var_explained[pc_y_idx]
            
            # Usar mais casas decimais para componentes com baixa variância
            x_pct = f'{x_variance:.3%}' if x_variance < 0.01 else f'{x_variance:.1%}'
            y_pct = f'{y_variance:.3%}' if y_variance < 0.01 else f'{y_variance:.1%}'
            
            x_label = f'{pc_x} ({x_pct} da variância)'
            y_label = f'{pc_y} ({y_pct} da variância)'
            
            # Avisar sobre componentes com variância muito baixa
            if x_variance < 0.001 or y_variance < 0.001:
                print(f"\n*** AVISO: Componentes com variância muito baixa ***")
                print(f"PC{pc_x_idx+1}: {x_variance:.6%}, PC{pc_y_idx+1}: {y_variance:.6%}")
                print(f"Considere usar componentes principais mais informativos (PC1, PC2, PC3)")
            print(f"Labels com variância:")
            print(f"  X: {x_label}")
            print(f"  Y: {y_label}")
        else:
            x_label = pc_x
            y_label = pc_y
            print(f"Labels sem variância:")
            print(f"  X: {x_label}")
            print(f"  Y: {y_label}")
            if not show_variance:
                print("  Motivo: show_variance = False")
            if not (pc_x_idx < len(var_explained)):
                print(f"  Motivo: pc_x_idx ({pc_x_idx}) >= len(var_explained) ({len(var_explained)})")
            if not (pc_y_idx < len(var_explained)):
                print(f"  Motivo: pc_y_idx ({pc_y_idx}) >= len(var_explained) ({len(var_explained)})")
        
        ax.set_xlabel(x_label, fontfamily=font_settings.get('axes', 'DejaVu Sans'))
        ax.set_ylabel(y_label, fontfamily=font_settings.get('axes', 'DejaVu Sans'))
        
        # DEBUG: Verificar se labels foram aplicados
        actual_xlabel = ax.get_xlabel()
        actual_ylabel = ax.get_ylabel()
        print(f"Labels aplicados:")
        print(f"  X esperado: '{x_label}' -> obtido: '{actual_xlabel}'")
        print(f"  Y esperado: '{y_label}' -> obtido: '{actual_ylabel}'")
        
        # Title with matrix type information
        matrix_info = "Correlação" if matrix_type == "Correlação" else "Covariância"
        ax.set_title(f'PCA Biplot (Matriz de {matrix_info})', 
                     fontfamily=font_settings.get('title', 'DejaVu Sans'))
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Add zero lines for reference
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # Ajustar limites do gráfico para componentes com baixa variância
        if max(abs(x_values).max(), abs(y_values).max()) < 1.0:
            # Para componentes pequenos, usar margens menores
            x_margin = (x_values.max() - x_values.min()) * 0.05 or 0.01
            y_margin = (y_values.max() - y_values.min()) * 0.05 or 0.01
            
            ax.set_xlim(x_values.min() - x_margin, x_values.max() + x_margin)
            ax.set_ylim(y_values.min() - y_margin, y_values.max() + y_margin)
            print(f"Ajustado zoom para componentes pequenos: xlim={ax.get_xlim()}, ylim={ax.get_ylim()}")

    def _generate_dendrogram(self, ax, data, font_settings):
        """Generate dendrogram."""
        # Get dendrogram options - use default values if widgets don't exist
        try:
            method = self.dendro_method_combo.currentText()
            metric = self.dendro_metric_combo.currentText()
            num_groups = self.dendro_groups_spin.value()
            color_groups = self.dendro_color_groups_check.isChecked()
        except AttributeError:
            method = 'ward'      # Default method
            metric = 'euclidean' # Default metric
            num_groups = 1       # Default: 1 group (all black)
            color_groups = False # Default: no coloring
        
        # Select numeric columns
        numeric_data = data.select_dtypes(include=np.number)
        if numeric_data.empty:
            raise ValueError("Nenhuma coluna numérica encontrada para dendrograma")
        
        # Prepare labels
        labels = data['Genótipo'].tolist() if 'Genótipo' in data.columns else None
        
        # Generate linkage matrix
        linkage_matrix = linkage(numeric_data.values, method=method, metric=metric)
        
        # Force refresh widget values
        if hasattr(self, 'dendro_groups_spin'):
            self.dendro_groups_spin.repaint()
        if hasattr(self, 'dendro_color_groups_check'):
            self.dendro_color_groups_check.repaint()
            
        # Re-read the values after refresh
        try:
            num_groups = self.dendro_groups_spin.value()
            color_groups = self.dendro_color_groups_check.isChecked()
        except AttributeError:
            pass
        
        # DEBUG: Print dendrogram settings
        print(f"\n=== DENDROGRAMA DEBUG ===")
        print(f"Widget exists - dendro_groups_spin: {hasattr(self, 'dendro_groups_spin')}")
        print(f"Widget exists - dendro_color_groups_check: {hasattr(self, 'dendro_color_groups_check')}")
        if hasattr(self, 'dendro_groups_spin'):
            print(f"SpinBox current value: {self.dendro_groups_spin.value()}")
            print(f"SpinBox range: {self.dendro_groups_spin.minimum()} - {self.dendro_groups_spin.maximum()}")
        if hasattr(self, 'dendro_color_groups_check'):
            print(f"Checkbox checked: {self.dendro_color_groups_check.isChecked()}")
        print(f"Color groups (final): {color_groups}")
        print(f"Num groups (final): {num_groups}")
        print(f"Linkage matrix shape: {linkage_matrix.shape}")
        
        # Create dendrogram with conditional coloring
        if color_groups and num_groups > 1:
            # Usar um threshold simples baseado na altura máxima
            max_distance = linkage_matrix[:, 2].max()
            # Calcular threshold como uma fração da altura máxima
            color_threshold = max_distance * (1 - (num_groups - 1) / len(linkage_matrix))
            
            print(f"Max distance: {max_distance}")
            print(f"Color threshold calculated: {color_threshold}")
            
            dendrogram(linkage_matrix, labels=labels, ax=ax, 
                      leaf_rotation=90, leaf_font_size=10,
                      color_threshold=color_threshold)
            print("Dendrograma gerado COM cores")
        else:
            # Default: all black
            print("Gerando dendrograma PRETO (padrão)")
            dendrogram(linkage_matrix, labels=labels, ax=ax, 
                      leaf_rotation=90, leaf_font_size=10,
                      color_threshold=0,  # This makes all branches black
                      above_threshold_color='black')
        
        ax.set_title(f'Dendrograma ({method}, {metric})', fontfamily=font_settings['title'])
        ax.set_xlabel('Genótipos', fontfamily=font_settings['axes'])
        ax.set_ylabel('Distância', fontfamily=font_settings['axes'])
        
        # Apply font to tick labels
        for label in ax.get_xticklabels():
            label.set_fontfamily(font_settings['tick'])

    def export_graph(self):
        """Export the current graph to a file with enhanced options."""
        if not self.current_graph_fig:
            QMessageBox.warning(self, "Aviso", "Nenhum gráfico disponível para exportar. Gere um gráfico primeiro.")
            return
        
        # Get export settings
        try:
            export_format = self.export_format_combo.currentText()
            dpi = self.export_dpi_spin.value()
        except AttributeError:
            export_format = 'PNG'
            dpi = 300
        
        # Define file extensions and filters
        format_filters = {
            'PNG': ('png', "PNG files (*.png)"),
            'PDF': ('pdf', "PDF files (*.pdf)"),
            'SVG': ('svg', "SVG files (*.svg)"),
            'JPG': ('jpg', "JPEG files (*.jpg)"),
            'EPS': ('eps', "EPS files (*.eps)"),
            'TIFF': ('tiff', "TIFF files (*.tiff)")
        }
        
        # Build filter string
        all_filters = ";;".join([f[1] for f in format_filters.values()])
        default_ext = format_filters.get(export_format, ('png', "PNG files (*.png)"))[0]
        default_name = f"grafico_ipastic.{default_ext}"
        
        # Open file dialog
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, 
            "Exportar Gráfico", 
            default_name,
            all_filters
        )
        
        if file_path:
            try:
                # Determine format from file extension if not specified
                if not file_path.lower().endswith(('.png', '.pdf', '.svg', '.jpg', '.jpeg', '.eps', '.tiff')):
                    file_path += f".{default_ext}"
                
                # Set export parameters based on format
                export_params = {
                    'dpi': dpi,
                    'bbox_inches': 'tight',
                    'facecolor': 'white',
                    'edgecolor': 'none',
                    'pad_inches': 0.1
                }
                
                # Format-specific optimizations
                file_ext = file_path.lower().split('.')[-1]
                if file_ext in ['jpg', 'jpeg']:
                    export_params['format'] = 'jpeg'
                    export_params['quality'] = 95
                elif file_ext == 'pdf':
                    export_params['format'] = 'pdf'
                    export_params['metadata'] = {
                        'Title': 'iPASTIC Analysis Graph',
                        'Creator': 'iPASTIC Desktop v2.5',
                        'Subject': 'Plant stress analysis visualization'
                    }
                elif file_ext == 'svg':
                    export_params['format'] = 'svg'
                    export_params['transparent'] = True
                elif file_ext == 'eps':
                    export_params['format'] = 'eps'
                elif file_ext == 'tiff':
                    export_params['format'] = 'tiff'
                    export_params['compression'] = 'lzw'
                
                # Save the figure
                self.current_graph_fig.savefig(file_path, **export_params)
                
                # Show success message with file info
                import os
                file_size = os.path.getsize(file_path) / 1024  # KB
                QMessageBox.information(self, "Sucesso", 
                    f"Gráfico exportado com sucesso!\n\n"
                    f"Arquivo: {file_path}\n"
                    f"Formato: {file_ext.upper()}\n"
                    f"DPI: {dpi}\n"
                    f"Tamanho: {file_size:.1f} KB")
                
                self.status_bar.showMessage(f"Gráfico salvo em {file_path}", 5000)
                
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao exportar gráfico:\n{str(e)}")
                logging.error(f"Graph export error: {e}", exc_info=True)

    def export_to_excel(self):
        """Exports the input data, indices, and rankings to a single Excel file."""
        if self.indices.empty:
            QMessageBox.warning(self, "Dados Ausentes", "Não há dados para exportar. Por favor, importe e calcule os índices primeiro.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Exportar Resultados", "resultados_ipastic.xlsx", "Excel Files (*.xlsx)")
        if not file_path:
            return
            
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Sheet 1: Input Data
                self.data.to_excel(writer, sheet_name='Dados de Entrada', index=False)
                # Sheet 2: Calculated Indices
                self.indices.to_excel(writer, sheet_name='Indices Calculados', index=False)
                # Sheet 3: Rankings
                self.rankings_df.to_excel(writer, sheet_name='Rankings', index=False)

            self.status_bar.showMessage(f"Resultados exportados com sucesso para {file_path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Erro de Exportação", f"Falha ao exportar para Excel.\nDetalhe: {e}")
            logging.error(f"Excel export error: {e}", exc_info=True)

    # --- UI Population and State Management ---

    def populate_table(self, table_widget, df):
        """Generic function to populate a QTableWidget from a DataFrame."""
        table_widget.clear()
        if df.empty:
            return

        headers = [str(h).replace('_', ' ').title() for h in df.columns]
        table_widget.setRowCount(len(df))
        table_widget.setColumnCount(len(df.columns))
        table_widget.setHorizontalHeaderLabels(headers)

        for i, row in df.iterrows():
            for j, col_name in enumerate(df.columns):
                value = row[col_name]
                item_text = f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)
                item = QTableWidgetItem(item_text)
                item.setTextAlignment(Qt.AlignCenter)
                
                # Adicionar valor numérico para ordenação correta se for número
                if isinstance(value, (float, int, np.floating, np.integer)):
                    item.setData(Qt.UserRole, float(value))

                # Coloração para rankings: busca cor nos índices
                if col_name == 'Genotipo' and hasattr(self, 'indices') and 'Cor' in self.indices.columns:
                    cor = self.indices.loc[self.indices['Genotipo'] == row['Genotipo'], 'Cor']
                    if not cor.empty:
                        item.setBackground(QColor(cor.values[0]))
                elif col_name.startswith('Rank_'):
                    position = row[col_name]
                    total = len(df)
                    if position <= total * 0.25: item.setBackground(QColor('#A5D6A7')) # Light Green
                    elif position <= total * 0.75: item.setBackground(QColor('#FFF59D')) # Light Yellow
                    else: item.setBackground(QColor('#EF9A9A')) # Light Red

                table_widget.setItem(i, j, item)

        table_widget.resizeColumnsToContents()

    def populate_input_table(self):
        """Populates the input table with standardized data."""
        self.input_table.clear()
        self.input_table.setRowCount(len(self.data))
        self.input_table.setColumnCount(3)
        self.input_table.setHorizontalHeaderLabels(self.original_column_names)
        for i, row in self.data.iterrows():
            #TODO:
            self.input_table.setItem(i, 0, QTableWidgetItem(str(row['Genotipo'])))
            self.input_table.setItem(i, 1, QTableWidgetItem(str(row['Yp'])))
            self.input_table.setItem(i, 2, QTableWidgetItem(str(row['Ys'])))
        self.input_table.resizeColumnsToContents()

    def populate_indices_table(self):
        """Popula a tabela de índices com cores para classificação por RP."""
        table_widget = self.indices_table
        df = self.indices.copy()
        
        table_widget.clear()
        if df.empty:
            return

        headers = [str(h).replace('_', ' ').title() for h in df.columns if h != 'Cor']
        table_widget.setRowCount(len(df))
        table_widget.setColumnCount(len(headers))
        table_widget.setHorizontalHeaderLabels(headers)

        for i, row in df.iterrows():
            col_index = 0  # Índice da coluna na tabela
            for j, col_name in enumerate(df.columns):
                if col_name == 'Cor':  # Pular coluna de cor
                    continue
                    
                value = row[col_name]
                item_text = f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)
                item = QTableWidgetItem(item_text)
                item.setTextAlignment(Qt.AlignCenter)

                # Coloração para rankings: busca cor nos índices
                if col_name == 'Genotipo' and hasattr(self, 'indices') and 'Cor' in self.indices.columns:
                    cor = self.indices.loc[self.indices['Genotipo'] == row['Genotipo'], 'Cor']
                    if not cor.empty:
                        item.setBackground(QColor(cor.values[0]))
                elif col_name == 'Classificação por RP' and 'Cor' in df.columns:
                    color = row['Cor']
                    item.setBackground(QColor(color))
                    # Texto branco para cores escuras
                    if color in ['#F44336', '#FF9800']:
                        item.setForeground(QColor('#FFFFFF'))
                elif col_name == 'RP' and 'Cor' in df.columns:
                    # Aplicar cor também na coluna RP
                    color = row['Cor']
                    # Usar uma versão mais suave da cor
                    light_colors = {
                        '#4CAF50': '#E8F5E8',    # Verde claro
                        '#8BC34A': '#F1F8E9',    # Verde muito claro
                        '#FF9800': '#FFF3E0',    # Laranja claro
                        '#F44336': '#FFEBEE'     # Vermelho claro
                    }
                    bg_color = light_colors.get(color, '#FFFFFF')
                    item.setBackground(QColor(bg_color))

                table_widget.setItem(i, col_index, item)
                col_index += 1  # Incrementar o índice da coluna na tabela

        table_widget.resizeColumnsToContents()

    def populate_rankings_table(self):
        """Popula a tabela de rankings com cores para classificação por RP."""
        table_widget = self.rank_table
        df = self.rankings_df.copy()
        
        table_widget.clear()
        if df.empty:
            return

        headers = [str(h).replace('_', ' ').title() for h in df.columns if h != 'Cor']
        table_widget.setRowCount(len(df))
        table_widget.setColumnCount(len(headers))
        table_widget.setHorizontalHeaderLabels(headers)

        for i, row in df.iterrows():
            col_index = 0  # Índice da coluna na tabela
            for j, col_name in enumerate(df.columns):
                if col_name == 'Cor':  # Pular coluna de cor
                    continue
                    
                value = row[col_name]
                item_text = f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)
                item = QTableWidgetItem(item_text)
                item.setTextAlignment(Qt.AlignCenter)

                # Aplicar cores baseadas na classificação por RP
                if col_name == 'Classificação por RP' and 'Cor' in df.columns:
                    color = row['Cor']
                    item.setBackground(QColor(color))
                    # Texto branco para cores escuras
                    if color in ['#F44336', '#FF9800']:
                        item.setForeground(QColor('#FFFFFF'))
                elif col_name.startswith('Rank_'):
                    # Manter a coloração original para rankings
                    position = row[col_name]
                    total = len(df)
                    if position <= total * 0.25: 
                        item.setBackground(QColor('#A5D6A7'))  # Light Green
                    elif position <= total * 0.75: 
                        item.setBackground(QColor('#FFF59D'))  # Light Yellow
                    else: 
                        item.setBackground(QColor('#EF9A9A'))  # Light Red

                table_widget.setItem(i, col_index, item)
                col_index += 1  # Incrementar o índice da coluna na tabela

        table_widget.resizeColumnsToContents()

    def populate_statistics_table(self, table_widget, stats_df, title=""):
        """Populates a table with descriptive statistics (index as first column)."""
        table_widget.clear()
        if stats_df.empty:
            return
        
        # Transpose so variables become rows
        stats_transposed = stats_df.T
        
        headers = ['Variável'] + list(stats_transposed.columns)
        table_widget.setRowCount(len(stats_transposed))
        table_widget.setColumnCount(len(headers))
        table_widget.setHorizontalHeaderLabels(headers)
        
        for i, (index_name, row) in enumerate(stats_transposed.iterrows()):
            # Clean variable names
            clean_name = index_name.replace('Rank_', '') if index_name.startswith('Rank_') else index_name
            table_widget.setItem(i, 0, QTableWidgetItem(clean_name))
            
            for j, value in enumerate(row):
                item_text = f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
                item = QTableWidgetItem(item_text)
                item.setTextAlignment(Qt.AlignCenter)
                table_widget.setItem(i, j + 1, item)

        
        table_widget.resizeColumnsToContents()
    
    def populate_correlation_table(self, table_widget, corr_matrix, title=""):
        """Populates a table with correlation matrix."""
        table_widget.clear()
        if corr_matrix.empty:
            return
        
        # Clean column names
        clean_columns = [col.replace('Rank_', '') if col.startswith('Rank_') else col for col in corr_matrix.columns]
        clean_index = [idx.replace('Rank_', '') if idx.startswith('Rank_') else idx for idx in corr_matrix.index]
        
        headers = ['Variável'] + clean_columns
        table_widget.setRowCount(len(corr_matrix))
        table_widget.setColumnCount(len(headers))
        table_widget.setHorizontalHeaderLabels(headers)
        
        for i, (index_name, row) in enumerate(corr_matrix.iterrows()):
            table_widget.setItem(i, 0, QTableWidgetItem(clean_index[i]))
            
            for j, value in enumerate(row):
                item_text = f"{value:.3f}" if not pd.isna(value) else "N/A"
                item = QTableWidgetItem(item_text)
                item.setTextAlignment(Qt.AlignCenter)
                
                # Color coding for correlation strength
                if not pd.isna(value):
                    abs_corr = abs(value)
                    if abs_corr >= 0.7:
                        item.setBackground(QColor('#ff9999'))  # Strong correlation - red
                    elif abs_corr >= 0.5:
                        item.setBackground(QColor('#ffcc99'))  # Moderate correlation - orange
                    elif abs_corr >= 0.3:
                        item.setBackground(QColor('#ffff99'))  # Weak correlation - yellow
                
                table_widget.setItem(i, j + 1, item)
        
        table_widget.resizeColumnsToContents()
    
    def populate_generic_table(self, table_widget, df, title=""):
        """Populates a table with generic DataFrame data."""
        table_widget.clear()
        if df.empty:
            return
        
        headers = list(df.columns)
        table_widget.setRowCount(len(df))
        table_widget.setColumnCount(len(df.columns))
        table_widget.setHorizontalHeaderLabels(headers)
        
        for i, row in df.iterrows():
            for j, (col_name, value) in enumerate(row.items()):
                if col_name == 'Variável' and isinstance(value, str) and value.startswith('Rank_'):
                    value = value.replace('Rank_', '')

                # Se for coluna de p-valor e não houver valor, mostrar 'N/A' ou explicação
                if 'p-valor' in col_name:
                    if value == '' or value is None or (isinstance(value, float) and pd.isna(value)):
                        # Anderson-Darling não retorna p-valor, mostrar explicação
                        if 'Anderson-Darling' in str(row.get('Teste', '')) or 'Anderson-Darling' in str(row.get('Teste', '')):
                            item_text = 'Não disponível (veja valor crítico)'
                        else:
                            item_text = 'N/A'
                    else:
                        item_text = f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)
                else:
                    item_text = f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)

                item = QTableWidgetItem(item_text)
                item.setTextAlignment(Qt.AlignCenter)

                # Color coding for p-values
                if 'p-valor' in col_name and isinstance(value, (float, np.floating)):
                    if value < 0.001:
                        item.setBackground(QColor('#ff6666'))  # Highly significant - red
                    elif value < 0.01:
                        item.setBackground(QColor('#ff9999'))  # Very significant - light red
                    elif value < 0.05:
                        item.setBackground(QColor('#ffcccc'))  # Significant - very light red

                table_widget.setItem(i, j, item)
        
        table_widget.resizeColumnsToContents()
    
    def clear_table(self, table_widget):
        """Clears a table widget."""
        table_widget.clear()
        table_widget.setRowCount(0)
        table_widget.setColumnCount(0)

    def get_data_from_table(self):
        """Reads data from the input table back into a DataFrame for processing."""
        rows = self.input_table.rowCount()
        cols = self.input_table.columnCount()
        data = []
        for row in range(rows):
            row_data = [self.input_table.item(row, col).text() if self.input_table.item(row, col) else '' for col in range(cols)]
            if any(d.strip() for d in row_data): # Only add if row is not empty
                data.append(row_data)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=self.original_column_names)
        # Re-run validation
        try:
            df_std, _ = self.data_manager.validate_and_standardize(df)
            return df_std
        except ValueError as e:
            QMessageBox.critical(self, "Erro de Validação", str(e))
            return pd.DataFrame()

    def check_data_validity(self):
        """Verifica se há pelo menos 3 linhas completas com dados válidos na tabela."""
        if not hasattr(self, 'input_table'):
            return False
        
        rows = self.input_table.rowCount()
        valid_rows = 0
        
        for row in range(rows):
            # Verificar se a linha tem dados nas três colunas essenciais: Genótipo, Yp, Ys
            genotype_item = self.input_table.item(row, 0)  # Coluna Genótipo
            yp_item = self.input_table.item(row, 1)        # Coluna Yp 
            ys_item = self.input_table.item(row, 2)        # Coluna Ys
            
            if (genotype_item and genotype_item.text().strip() and
                yp_item and yp_item.text().strip() and
                ys_item and ys_item.text().strip()):
                
                # Verificar se Yp e Ys são números válidos
                try:
                    yp_val = float(yp_item.text().strip())
                    ys_val = float(ys_item.text().strip())
                    if yp_val > 0 and ys_val > 0:  # Valores devem ser positivos
                        valid_rows += 1
                except (ValueError, TypeError):
                    continue
        
        return valid_rows >= 3

    def on_table_data_changed(self, item):
        """Função chamada quando dados na tabela são alterados."""
        # Atualizar o estado da UI quando dados mudam
        self.update_ui_state()

    def update_ui_state(self):
        """Enables or disables UI elements based on the application's state."""
        data_is_present = not self.data.empty
        data_is_valid = self.check_data_validity()  # Verificar se há 3 linhas válidas
        indices_are_calculated = not self.indices.empty
        clustering_is_done = 'Cluster' in self.indices.columns

        # Actions and buttons
        self.calc_action.setEnabled(data_is_valid)  # Usar validação mais rigorosa
        self.export_excel_action.setEnabled(indices_are_calculated)
        self.export_graph_button.setEnabled(self.current_graph_fig is not None)
        self.clear_action.setEnabled(data_is_present)

        # Table control buttons (always enabled for manual data entry)
        if hasattr(self, 'add_row_btn'):
            self.add_row_btn.setEnabled(True)
        if hasattr(self, 'remove_row_btn'):
            self.remove_row_btn.setEnabled(True)
        if hasattr(self, 'clear_cell_btn'):
            self.clear_cell_btn.setEnabled(True)
        
        # Statistics and clustering controls
        if hasattr(self, 'cluster_button'):
            self.cluster_button.setEnabled(indices_are_calculated)
        if hasattr(self, 'stats_use_rankings'):
            self.stats_use_rankings.setEnabled(not self.rankings_df.empty)
        if hasattr(self, 'refresh_stats_btn'):
            self.refresh_stats_btn.setEnabled(indices_are_calculated)

        # Tabs
        self.central_widget.setTabEnabled(1, indices_are_calculated) # Indices
        self.central_widget.setTabEnabled(2, indices_are_calculated) # Rankings
        self.central_widget.setTabEnabled(3, indices_are_calculated) # Statistics
        self.central_widget.setTabEnabled(4, indices_are_calculated) # Graphics
        self.central_widget.setTabEnabled(5, indices_are_calculated) # PCA Results
        self.central_widget.setTabEnabled(6, indices_are_calculated) # Regression
        self.central_widget.setTabEnabled(7, indices_are_calculated) # Reports

        # Graphics controls
        self.generate_graph_button.setEnabled(indices_are_calculated)
        self.export_graph_button.setEnabled(self.current_graph_fig is not None)
        # Enable hue combo when indices are calculated (for classification) or clustering is done
        if hasattr(self, 'hue_combo'):
            self.hue_combo.setEnabled(indices_are_calculated)
        
        # Reports controls
        if hasattr(self, 'generate_report_btn'):
            self.generate_report_btn.setEnabled(indices_are_calculated)
        
        # Regression controls
        if hasattr(self, 'update_regression_vars_btn'):
            self.update_regression_vars_btn.setEnabled(indices_are_calculated)
        if hasattr(self, 'run_regression_btn'):
            self.run_regression_btn.setEnabled(indices_are_calculated)

    # --- Event Handlers & Settings ---

    def run_in_worker(self, func, *args, on_finish, **kwargs):
        """Manages running a function in a background thread."""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Aguarde", "Outra operação já está em andamento. Por favor, aguarde a sua conclusão.")
            return
        self.worker = Worker(func, *args, **kwargs)
        self.worker.finished.connect(on_finish)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_worker_done)
        self.worker.error.connect(self.on_worker_done)
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage("Processando, por favor aguarde...")
        self.worker.start()

    def on_worker_error(self, error_message):
        QMessageBox.critical(self, "Erro na Operação", error_message)

    def on_worker_done(self):
        self.progress_bar.setVisible(False)
        self.status_bar.clearMessage()
        self.worker = None

    # Funções de copy/paste para tabelas de resultados
    def copy_table_selection(self, table):
        """Copia a seleção da tabela para a área de transferência."""
        try:
            selection = table.selectionModel().selectedIndexes()
            if not selection:
                # Se não há seleção, copiar tudo
                rows = list(range(table.rowCount()))
                cols = list(range(table.columnCount()))
            else:
                # Organizar seleção por linha e coluna
                rows = sorted(set(index.row() for index in selection))
                cols = sorted(set(index.column() for index in selection))
            
            # Construir texto da seleção
            clipboard_text = []
            for row in rows:
                row_data = []
                for col in cols:
                    item = table.item(row, col)
                    cell_text = item.text() if item else ""
                    row_data.append(cell_text)
                clipboard_text.append("\t".join(row_data))
            
            # Copiar para área de transferência
            if clipboard_text:
                QApplication.clipboard().setText("\n".join(clipboard_text))
        except Exception as e:
            print(f"Erro ao copiar: {e}")
            QMessageBox.warning(table, "Erro", f"Erro ao copiar dados: {e}")

    def copy_table_with_headers(self, table):
        """Copia a seleção da tabela incluindo cabeçalhos."""
        try:
            selection = table.selectionModel().selectedIndexes()
            if not selection:
                # Se não há seleção, copiar tudo
                rows = list(range(table.rowCount()))
                cols = list(range(table.columnCount()))
            else:
                # Organizar seleção por linha e coluna
                rows = sorted(set(index.row() for index in selection))
                cols = sorted(set(index.column() for index in selection))
            
            # Adicionar cabeçalhos de coluna
            headers = []
            for col in cols:
                header_item = table.horizontalHeaderItem(col)
                header_text = header_item.text() if header_item else f"Col{col}"
                headers.append(header_text)
            
            clipboard_text = ["\t".join(headers)]
            
            # Adicionar dados das células
            for row in rows:
                row_data = []
                for col in cols:
                    item = table.item(row, col)
                    cell_text = item.text() if item else ""
                    row_data.append(cell_text)
                clipboard_text.append("\t".join(row_data))
            
            # Copiar para área de transferência
            if clipboard_text:
                QApplication.clipboard().setText("\n".join(clipboard_text))
        except Exception as e:
            print(f"Erro ao copiar com cabeçalho: {e}")
            QMessageBox.warning(table, "Erro", f"Erro ao copiar dados com cabeçalho: {e}")

    def show_result_table_context_menu(self, position, table):
        """Mostra menu de contexto para tabelas de resultado."""
        try:
            # Verificar se a tabela tem dados
            if table.rowCount() == 0 or table.columnCount() == 0:
                return
            
            menu = QMenu(table)
            
            copy_action = menu.addAction("Copiar")
            copy_action.setShortcut(QKeySequence.Copy)
            copy_action.triggered.connect(lambda: self.copy_table_selection(table))
            
            copy_with_headers_action = menu.addAction("Copiar com Cabeçalho")
            copy_with_headers_action.triggered.connect(lambda: self.copy_table_with_headers(table))
            
            menu.exec_(table.mapToGlobal(position))
        except Exception as e:
            print(f"Erro no menu de contexto: {e}")
            # Não mostrar erro para o usuário, apenas log

    def update_regression_variables(self):
        """Atualiza as variáveis disponíveis para regressão."""
        if self.indices.empty:
            self.regression_status_label.setText("⚠️ Calcule os índices primeiro")
            return
        
        # Obter variáveis numéricas
        numeric_vars = self.indices.select_dtypes(include=np.number).columns.tolist()
        # Remover colunas não relevantes
        exclude_vars = ['Cor', 'Cluster']
        numeric_vars = [var for var in numeric_vars if var not in exclude_vars]
        
        # Atualizar ComboBox da variável dependente
        self.regression_y_var.clear()
        self.regression_y_var.addItems(numeric_vars)
        
        # Atualizar lista de variáveis independentes
        self.regression_x_vars.clear()
        for var in numeric_vars:
            self.regression_x_vars.addItem(var)
        
        self.regression_status_label.setText(f"✅ {len(numeric_vars)} variáveis disponíveis")

    def run_regression_analysis(self):
        """Executa a análise de regressão múltipla."""
        if self.indices.empty:
            QMessageBox.warning(self, "Dados Ausentes", 
                              "É necessário calcular os índices antes de executar a regressão.")
            return
        
        # Obter variável dependente
        y_var = self.regression_y_var.currentText()
        if not y_var:
            QMessageBox.warning(self, "Variável Dependente", 
                              "Selecione uma variável dependente.")
            return
        
        # Obter variáveis independentes selecionadas
        selected_items = self.regression_x_vars.selectedItems()
        if len(selected_items) < 1:
            QMessageBox.warning(self, "Variáveis Independentes", 
                              "Selecione pelo menos uma variável independente.")
            return
        
        x_vars = [item.text() for item in selected_items]
        
        # Verificar se a variável dependente não está nas independentes
        if y_var in x_vars:
            QMessageBox.warning(self, "Variáveis Conflitantes", 
                              "A variável dependente não pode estar entre as independentes.")
            return
        
        try:
            # Executar regressão em worker thread
            self.run_in_worker(
                self._perform_regression_analysis,
                y_var, x_vars, 
                self.include_intercept.isChecked(),
                self.standardize_vars.isChecked(),
                self.show_residuals.isChecked(),
                on_finish=self.on_regression_finished
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Erro na Regressão", f"Erro ao executar regressão:\n{str(e)}")
            logging.error(f"Regression error: {e}", exc_info=True)

    def _perform_regression_analysis(self, y_var, x_vars, include_intercept, standardize, show_residuals):
        """Executa a análise de regressão (método para worker thread)."""
        #TODO:FIX IMPORTS
        try:
            import numpy as np
            import pandas as pd
            from scipy import stats
            import statsmodels.api as sm
            from sklearn.preprocessing import StandardScaler
            
            # Preparar dados incluindo coluna de genótipos se disponível
            cols_to_include = [y_var] + x_vars
            
            # Verificar se existe uma coluna de genótipos e incluí-la
            possible_genotype_cols = ['Genótipo', 'Genotype', 'ID', 'Nome', 'Name', 'Genotipo']
            genotype_col = None
            for col in possible_genotype_cols:
                if col in self.indices.columns:
                    genotype_col = col
                    if col not in cols_to_include:
                        cols_to_include.append(col)
                    break
            
            data_subset = self.indices[cols_to_include].dropna()
            
            print(f"DEBUG: Incluindo coluna de genótipos '{genotype_col}' nos dados da regressão")
            print(f"DEBUG: Colunas em data_subset: {list(data_subset.columns)}")
            print(f"DEBUG: Ordem preservada - primeiros genótipos: {data_subset[genotype_col].head().tolist() if genotype_col else 'N/A'}")
            
            if len(data_subset) < len(x_vars) + 2:
                raise ValueError(f"Dados insuficientes para regressão. Necessário pelo menos {len(x_vars) + 2} observações, encontrado {len(data_subset)}")
            
            y = data_subset[y_var].values
            X = data_subset[x_vars].values
            
            # Padronizar se solicitado
            if standardize:
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                X_scaled = scaler_X.fit_transform(X)
                y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
                X_reg = X_scaled
                y_reg = y_scaled
            else:
                X_reg = X
                y_reg = y
            
            # Adicionar intercepto se solicitado
            if include_intercept:
                X_reg = sm.add_constant(X_reg)
                var_names = ['Intercepto'] + x_vars
            else:
                var_names = x_vars
            
            # Ajustar modelo
            model = sm.OLS(y_reg, X_reg).fit()
            
            # Calcular métricas adicionais
            y_pred = model.predict(X_reg)
            residuals = y_reg - y_pred
            
            # Testes de diagnóstico
            from statsmodels.stats.diagnostic import het_breuschpagan, het_white
            from statsmodels.stats.stattools import durbin_watson, jarque_bera
            
            # Teste de homocedasticidade (Breusch-Pagan)
            try:
                bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X_reg)
            except:
                bp_stat, bp_pvalue = np.nan, np.nan
            
            # Teste de normalidade dos resíduos (Jarque-Bera)
            try:
                jb_stat, jb_pvalue, _, _ = jarque_bera(residuals)
            except:
                jb_stat, jb_pvalue = np.nan, np.nan
            
            # Estatística Durbin-Watson (autocorrelação)
            try:
                dw_stat = durbin_watson(residuals)
            except:
                dw_stat = np.nan
            
            # VIF (Variance Inflation Factor) para multicolinearidade
            vif_values = []
            if len(x_vars) > 1 and include_intercept:
                try:
                    from statsmodels.stats.outliers_influence import variance_inflation_factor
                    # VIF calculado apenas para variáveis (sem intercepto)
                    X_vif = X_reg[:, 1:] if include_intercept else X_reg
                    for i in range(X_vif.shape[1]):
                        vif = variance_inflation_factor(X_vif, i)
                        vif_values.append(vif)
                except:
                    vif_values = [np.nan] * len(x_vars)
            else:
                vif_values = [np.nan] * len(x_vars)
            
            # Análise de resíduos se solicitado
            residual_analysis = None
            if show_residuals:
                residual_analysis = {
                    'residuals': residuals,
                    'fitted_values': y_pred,
                    'standardized_residuals': residuals / np.std(residuals),
                    'studentized_residuals': residuals / (np.std(residuals) * np.sqrt(1 - np.diag(X_reg @ np.linalg.pinv(X_reg.T @ X_reg) @ X_reg.T))),
                }
            
            return {
                'model': model,
                'y_var': y_var,
                'x_vars': x_vars,
                'var_names': var_names,
                'include_intercept': include_intercept,
                'standardized': standardize,
                'n_obs': len(data_subset),
                'bp_test': (bp_stat, bp_pvalue),
                'jb_test': (jb_stat, jb_pvalue),
                'dw_stat': dw_stat,
                'vif_values': vif_values,
                'residual_analysis': residual_analysis,
                'data_used': data_subset,
                'data_copy': data_subset,
                'genotype_col': genotype_col
            }
            
        except Exception as e:
            raise Exception(f"Erro na análise de regressão: {str(e)}")

    def on_regression_finished(self, results):
        """Processa os resultados da regressão e popula as abas."""
        try:
            model = results['model']
            
            # === ABA 1: RESUMO DO MODELO ===
            
            # Gerar equação do modelo
            def generate_model_equation(model, y_var, x_vars, include_intercept):
                """Gera a equação do modelo de regressão."""
                try:
                    print(f"\n=== DEBUG GERAÇÃO EQUAÇÃO ===")
                    print(f"y_var: {y_var}")
                    print(f"x_vars: {x_vars}")
                    print(f"include_intercept: {include_intercept}")
                    print(f"Tipo do model: {type(model)}")
                    print(f"Atributos do model: {[attr for attr in dir(model) if not attr.startswith('_')]}")
                    
                    # Obter coeficientes
                    params = model.params
                    print(f"Tipo de params: {type(params)}")
                    print(f"params: {params}")
                    print(f"len(params): {len(params)}")
                    
                    # Abordagem mais direta - tentar diferentes formas de acessar os dados
                    param_values = {}
                    param_names = []
                    
                    # Método 1: Se é pandas Series
                    if hasattr(params, 'index'):
                        print("DEBUG: params é pandas Series")
                        param_names = list(params.index)
                        param_values = dict(params)
                    
                    # Método 2: Se é numpy array, tentar obter nomes do modelo
                    else:
                        print("DEBUG: params é numpy array")
                        
                        # Buscar nomes em vários lugares possíveis
                        possible_names = []
                        
                        if hasattr(model, 'model') and hasattr(model.model, 'exog_names'):
                            possible_names = list(model.model.exog_names)
                            print(f"DEBUG: Nomes de model.model.exog_names: {possible_names}")
                        
                        if hasattr(model, 'exog_names'):
                            possible_names = list(model.exog_names)
                            print(f"DEBUG: Nomes de model.exog_names: {possible_names}")
                        
                        # Se não encontrou nomes, criar baseado nas variáveis fornecidas
                        if not possible_names:
                            print("DEBUG: Criando nomes baseados em x_vars")
                            if include_intercept:
                                possible_names = ['const'] + list(x_vars)
                            else:
                                possible_names = list(x_vars)
                        
                        print(f"DEBUG: possible_names antes do corte: {possible_names}")
                        print(f"DEBUG: len(params): {len(params)}")
                        
                        # Forçar o uso dos nomes reais das variáveis
                        if include_intercept and len(params) == len(x_vars) + 1:
                            # Tem intercepto + variáveis
                            param_names = ['const'] + list(x_vars)
                        elif not include_intercept and len(params) == len(x_vars):
                            # Apenas variáveis, sem intercepto
                            param_names = list(x_vars)
                        else:
                            # Usar os nomes encontrados, mas limitar ao tamanho dos parâmetros
                            param_names = possible_names[:len(params)]
                        
                        print(f"DEBUG: param_names finais: {param_names}")
                        
                        # Criar dicionário
                        for i, name in enumerate(param_names):
                            if i < len(params):
                                param_values[name] = float(params[i])
                                print(f"DEBUG: Mapeando params[{i}] = {params[i]} -> {name}")
                    
                    print(f"DEBUG FINAL:")
                    print(f"  param_names: {param_names}")
                    print(f"  param_values: {param_values}")
                    print(f"  x_vars fornecidas: {x_vars}")
                    print(f"  include_intercept: {include_intercept}")
                    
                    if len(param_values) == 0:
                        return f"{y_var} = [Modelo sem coeficientes detectados]"
                    
                    # Abordagem super simples - apenas usar todos os parâmetros disponíveis
                    print(f"\n=== CONSTRUÇÃO SIMPLES DA EQUAÇÃO ===")
                    
                    equation_parts = []
                    
                    # Verificar quais chaves existem em param_values
                    print(f"DEBUG: Chaves disponíveis em param_values: {list(param_values.keys())}")
                    
                    # Tentar encontrar intercepto primeiro
                    intercepto_encontrado = None
                    for key in ['const', 'Intercept', 'intercept', 'constant']:
                        if key in param_values:
                            intercepto_encontrado = key
                            break
                    
                    if intercepto_encontrado and include_intercept:
                        intercept_val = param_values[intercepto_encontrado]
                        equation_parts.append(f"{intercept_val:.4f}")
                        print(f"DEBUG: Intercepto {intercepto_encontrado} = {intercept_val:.4f} adicionado")
                    
                    # Adicionar todas as outras variáveis que não são intercepto
                    for key, value in param_values.items():
                        # Pular intercepto
                        if key in ['const', 'Intercept', 'intercept', 'constant']:
                            continue
                        
                        print(f"DEBUG: Examinando variável {key} = {value}")
                        print(f"DEBUG: {key} está em x_vars? {key in x_vars}")
                        
                        # Adicionar termo
                        if len(equation_parts) > 0:  # Já tem outros termos
                            if value >= 0:
                                term = f"+ {value:.4f}*{key}"
                            else:
                                term = f"- {abs(value):.4f}*{key}"
                        else:  # É o primeiro termo
                            term = f"{value:.4f}*{key}"
                        
                        equation_parts.append(term)
                        print(f"DEBUG: Termo adicionado: '{term}'")
                    
                    print(f"DEBUG: Lista final de partes: {equation_parts}")
                    print(f"DEBUG: Número de partes: {len(equation_parts)}")
                    
                    # Construir equação
                    if equation_parts:
                        equation = f"{y_var} = " + " ".join(equation_parts)
                        print(f"DEBUG: *** EQUAÇÃO FINAL: '{equation}' ***")
                        return equation
                    else:
                        return f"{y_var} = [ERRO: Nenhuma parte foi construída]"
                
                except Exception as e:
                    print(f"DEBUG: Erro na geração da equação: {e}")
                    import traceback
                    traceback.print_exc()
                    return f"Erro ao gerar equação: {str(e)}"
            
            model_equation = generate_model_equation(model, results['y_var'], results['x_vars'], results['include_intercept'])
            
            summary_text = f"""RESUMO DA REGRESSÃO MÚLTIPLA
========================================

Variável Dependente: {results['y_var']}
Variáveis Independentes: {', '.join(results['x_vars'])}

EQUAÇÃO DO MODELO:
{model_equation}

INFORMAÇÕES GERAIS:
Número de observações: {results['n_obs']}
Método: Mínimos Quadrados Ordinários (OLS)
Intercepto incluído: {'Sim' if results['include_intercept'] else 'Não'}
Variáveis padronizadas: {'Sim' if results['standardized'] else 'Não'}

ESTATÍSTICAS DO MODELO:
R-quadrado: {model.rsquared:.4f}
R-quadrado ajustado: {model.rsquared_adj:.4f}
F-estatística: {model.fvalue:.4f}
P-valor (F): {model.f_pvalue:.4e}
AIC: {model.aic:.4f}
BIC: {model.bic:.4f}
Log-Likelihood: {model.llf:.4f}

TESTES DE DIAGNÓSTICO:
Breusch-Pagan (Homocedasticidade):
  Estatística: {results['bp_test'][0]:.4f}
  P-valor: {results['bp_test'][1]:.4e}
  Interpretação: {'Homocedasticidade' if results['bp_test'][1] > 0.05 else 'Heterocedasticidade'}

Jarque-Bera (Normalidade dos resíduos):
  Estatística: {results['jb_test'][0]:.4f}
  P-valor: {results['jb_test'][1]:.4e}
  Interpretação: {'Resíduos normais' if results['jb_test'][1] > 0.05 else 'Resíduos não-normais'}

Durbin-Watson (Autocorrelação):
  Estatística: {results['dw_stat']:.4f}
  Interpretação: {'Sem autocorrelação' if 1.5 <= results['dw_stat'] <= 2.5 else 'Possível autocorrelação'}

INTERPRETAÇÃO:
- R² = {model.rsquared:.1%}: {model.rsquared*100:.1f}% da variação em {results['y_var']} é explicada pelo modelo
- F-test: {'Modelo significativo' if model.f_pvalue < 0.05 else 'Modelo não significativo'} (α = 0.05)
"""
            
            self.regression_summary_text.setText(summary_text)
            
            # === ABA 2: COEFICIENTES ===
            coef_data = []
            conf_intervals = model.conf_int()
            for i, var_name in enumerate(results['var_names']):
                coef = model.params[i]
                std_err = model.bse[i]
                t_stat = model.tvalues[i]
                p_value = model.pvalues[i]
                
                # Tratar tanto DataFrame quanto numpy array
                if hasattr(conf_intervals, 'iloc'):
                    ci_lower, ci_upper = conf_intervals.iloc[i, 0], conf_intervals.iloc[i, 1]
                else:
                    ci_lower, ci_upper = conf_intervals[i, 0], conf_intervals[i, 1]
                
                # VIF apenas para variáveis (não intercepto)
                if var_name == 'Intercepto':
                    vif = 'N/A'
                else:
                    var_idx = results['x_vars'].index(var_name) if var_name in results['x_vars'] else 0
                    vif_val = results['vif_values'][var_idx] if var_idx < len(results['vif_values']) else np.nan
                    vif = f"{vif_val:.2f}" if not np.isnan(vif_val) else 'N/A'
                
                significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                
                coef_data.append([
                    var_name,
                    f"{coef:.6f}",
                    f"{std_err:.6f}",
                    f"{t_stat:.4f}",
                    f"{p_value:.4e}",
                    f"[{ci_lower:.4f}, {ci_upper:.4f}]",
                    vif,
                    significance
                ])
            
            self.regression_coef_table.clear()
            self.regression_coef_table.setRowCount(len(coef_data))
            self.regression_coef_table.setColumnCount(8)
            self.regression_coef_table.setHorizontalHeaderLabels([
                'Variável', 'Coeficiente', 'Erro Padrão', 't-stat', 
                'P-valor', 'IC 95%', 'VIF', 'Signif.'
            ])
            
            for i, row in enumerate(coef_data):
                for j, value in enumerate(row):
                    self.regression_coef_table.setItem(i, j, QTableWidgetItem(str(value)))
            
            self.regression_coef_table.resizeColumnsToContents()
            
            # === ABA 3: DIAGNÓSTICOS ===
            diag_data = [
                ['Número de observações', str(results['n_obs'])],
                ['Graus de liberdade (modelo)', str(model.df_model)],
                ['Graus de liberdade (resíduos)', str(model.df_resid)],
                ['R-quadrado', f"{model.rsquared:.6f}"],
                ['R-quadrado ajustado', f"{model.rsquared_adj:.6f}"],
                ['Raiz do MSE', f"{np.sqrt(model.mse_resid):.6f}"],
                ['AIC', f"{model.aic:.4f}"],
                ['BIC', f"{model.bic:.4f}"],
                ['Log-Likelihood', f"{model.llf:.4f}"],
                ['F-estatística', f"{model.fvalue:.4f}"],
                ['P-valor (F)', f"{model.f_pvalue:.4e}"],
                ['Condição do modelo', f"{model.condition_number:.2f}"],
                ['Breusch-Pagan (estat.)', f"{results['bp_test'][0]:.4f}"],
                ['Breusch-Pagan (p-valor)', f"{results['bp_test'][1]:.4e}"],
                ['Jarque-Bera (estat.)', f"{results['jb_test'][0]:.4f}"],
                ['Jarque-Bera (p-valor)', f"{results['jb_test'][1]:.4e}"],
                ['Durbin-Watson', f"{results['dw_stat']:.4f}"]
            ]
            
            self.regression_diag_table.clear()
            self.regression_diag_table.setRowCount(len(diag_data))
            self.regression_diag_table.setColumnCount(2)
            self.regression_diag_table.setHorizontalHeaderLabels(['Métrica', 'Valor'])
            
            for i, (metric, value) in enumerate(diag_data):
                self.regression_diag_table.setItem(i, 0, QTableWidgetItem(metric))
                self.regression_diag_table.setItem(i, 1, QTableWidgetItem(value))
            
            self.regression_diag_table.resizeColumnsToContents()
            
            # === ABA 4: ANÁLISE DE RESÍDUOS ===
            if results['residual_analysis']:
                res_analysis = results['residual_analysis']
                residuals_text = f"""ANÁLISE DE RESÍDUOS
========================================

ESTATÍSTICAS DESCRITIVAS DOS RESÍDUOS:
Média: {np.mean(res_analysis['residuals']):.6f}
Desvio Padrão: {np.std(res_analysis['residuals']):.6f}
Mínimo: {np.min(res_analysis['residuals']):.6f}
Máximo: {np.max(res_analysis['residuals']):.6f}
Q1: {np.percentile(res_analysis['residuals'], 25):.6f}
Mediana: {np.percentile(res_analysis['residuals'], 50):.6f}
Q3: {np.percentile(res_analysis['residuals'], 75):.6f}

OUTLIERS NOS RESÍDUOS (|resíduo| > 2σ):
"""
                
                outliers = np.abs(res_analysis['standardized_residuals']) > 2
                if np.any(outliers):
                    outlier_indices = np.where(outliers)[0]
                    residuals_text += f"Encontrados {len(outlier_indices)} outliers:\n"
                    
                    # Obter nomes dos genótipos usando a mesma lógica da tabela de resíduos
                    data_used = results['data_used']
                    genotypes = []
                    possible_genotype_cols = ['Genótipo', 'Genotype', 'ID', 'Nome', 'Name', 'Genotipo']
                    genotype_col_found = None
                    
                    # Procurar por colunas de genótipos nos dados
                    for col in possible_genotype_cols:
                        if col in data_used.columns:
                            genotype_col_found = col
                            genotypes = data_used[col].tolist()
                            break
                    
                    # Se não encontrou uma coluna específica, tentar usar o índice
                    if not genotypes:
                        if hasattr(data_used, 'index') and not data_used.index.equals(pd.RangeIndex(len(data_used))):
                            genotypes = data_used.index.tolist()
                        else:
                            # Fallback: usar numeração sequencial simples
                            genotypes = [f'{i+1}' for i in range(len(data_used))]
                    
                    for idx in outlier_indices:  # Mostrar todos os outliers
                        genotype_name = str(genotypes[idx]) if idx < len(genotypes) else f'Observação {idx+1}'
                        residual_normal = res_analysis['residuals'][idx]
                        residual_std = res_analysis['standardized_residuals'][idx]
                        residuals_text += f"  {genotype_name}: resíduo = {residual_normal:.4f} (padronizado = {residual_std:.4f})\n"
                else:
                    residuals_text += "Nenhum outlier encontrado.\n"
                
                residuals_text += f"""
RECOMENDAÇÕES:
- Verifique gráficos de resíduos vs valores ajustados para padrões
- Analise resíduos padronizados para identificar outliers
- Considere transformações se houver heterocedasticidade
- Remova ou investigue outliers se apropriado
"""
                
                self.regression_residuals_text.setText(residuals_text)
            else:
                self.regression_residuals_text.setText("Análise de resíduos não solicitada.")
            
            # === TABELA DE VALORES DETALHADOS ===
            self.populate_residuals_values_table(results)
            
            # Status final
            r2_pct = model.rsquared * 100
            # Armazenar equação nos resultados
            results['equation'] = model_equation
            
            # Armazenar resultados para usar nos relatórios
            self.current_regression_results = results
            
            self.regression_status_label.setText(
                f"✅ Regressão concluída! R² = {r2_pct:.1f}% | "
                f"F = {model.fvalue:.2f} (p = {model.f_pvalue:.4f}) | "
                f"N = {results['n_obs']}"
            )
            self.status_bar.showMessage("Análise de regressão concluída!", 5000)
            
        except Exception as e:
            QMessageBox.critical(self, "Erro nos Resultados", 
                               f"Erro ao processar resultados da regressão:\n{str(e)}")
            logging.error(f"Regression results error: {e}", exc_info=True)
            self.regression_status_label.setText("❌ Erro ao processar resultados da regressão")

    def populate_residuals_values_table(self, results):
        """Popula a tabela com valores observados, estimados e resíduos de todos os genótipos."""
        try:
            # Imports necessários
            import numpy as np
            from PyQt5.QtWidgets import QTableWidgetItem
            from PyQt5.QtGui import QColor
            from PyQt5.QtCore import Qt
            
            # Verificar se a tabela existe
            if not hasattr(self, 'residuals_values_table'):
                print("Erro: Tabela residuals_values_table não encontrada")
                return
                
            model = results['model']
            data_used = results['data_used']
            
            # Obter dados com verificações de segurança
            y_var = results['y_var']
            observed_values = data_used[y_var]
            
            # Converter para array numpy se necessário
            if hasattr(observed_values, 'values'):
                observed_values = observed_values.values
            observed_values = np.asarray(observed_values)
            
            # Obter fitted values
            fitted_values = model.fittedvalues
            if hasattr(fitted_values, 'values'):
                fitted_values = fitted_values.values
            fitted_values = np.asarray(fitted_values)
                
            # Obter resíduos
            residuals = model.resid
            if hasattr(residuals, 'values'):
                residuals = residuals.values
            residuals = np.asarray(residuals)
            
            # Obter nomes dos genótipos usando a mesma lógica da PCA
            genotypes = []
            possible_genotype_cols = ['Genótipo', 'Genotype', 'ID', 'Nome', 'Name', 'Genotipo']
            genotype_col_found = None
            
            # Procurar por colunas de genótipos nos dados
            for col in possible_genotype_cols:
                if col in data_used.columns:
                    genotype_col_found = col
                    genotypes = data_used[col].tolist()
                    break
            
            # Se não encontrou uma coluna específica, tentar usar o índice
            if not genotypes:
                if hasattr(data_used, 'index') and not data_used.index.equals(pd.RangeIndex(len(data_used))):
                    genotypes = data_used.index.tolist()
                else:
                    # Fallback: usar numeração sequencial simples
                    genotypes = [f'{i+1}' for i in range(len(data_used))]
            
            print(f"DEBUG Resíduos: Colunas disponíveis em data_used: {list(data_used.columns)}")
            print(f"DEBUG Resíduos: Coluna de genótipos encontrada: {genotype_col_found}")
            print(f"DEBUG Resíduos: Usando nomes de genótipos: {genotypes[:5]}... (primeiros 5)")
            print(f"DEBUG Resíduos: Total de genótipos: {len(genotypes)}")
            
            genotypes = np.asarray([str(g) for g in genotypes])
            
            # Verificar tipos e formas dos dados
            print(f"Tipos de dados: obs={type(observed_values)}, fitted={type(fitted_values)}, resid={type(residuals)}, genot={type(genotypes)}")
            
            # Verificar se todos os arrays têm o mesmo comprimento
            n_rows = len(observed_values)
            if len(fitted_values) != n_rows or len(residuals) != n_rows or len(genotypes) != n_rows:
                print(f"Erro: Arrays com tamanhos diferentes - obs:{len(observed_values)}, fitted:{len(fitted_values)}, resid:{len(residuals)}, genot:{len(genotypes)}")
                return
            
            print(f"Populando tabela com {n_rows} linhas")  # Debug
            print(f"DEBUG: Genótipos a serem processados: {genotypes[:10]}...")  # Debug
            
            # Configurar tabela - Limpeza completa e robusta
            self.residuals_values_table.clear()
            self.residuals_values_table.setRowCount(n_rows)
            self.residuals_values_table.setColumnCount(6)
            
            # Headers da tabela
            headers = [
                'Genótipo', 
                'Valor Observado', 
                'Valor Estimado', 
                'Resíduo', 
                'Resíduo %',
                'Resíduo Abs'
            ]
            self.residuals_values_table.setHorizontalHeaderLabels(headers)
            
            print("Headers definidos")  # Debug
            
            # Calcular desvio padrão dos resíduos uma única vez (fora do loop)
            std_residuals = np.std(residuals)
            
            # Preencher dados
            successful_rows = 0
            for i in range(n_rows):
                try:
                    genotype = str(genotypes[i])
                    observed = float(observed_values[i])
                    fitted = float(fitted_values[i])
                    residual = float(residuals[i])
                    residual_pct = (residual / observed * 100) if observed != 0 else 0
                    residual_abs = abs(residual)
                    
                    # Debug para primeiro e último genótipo
                    if i in [0, n_rows-1]:
                        print(f"DEBUG linha {i}: {genotype} - obs:{observed:.4f}, fitted:{fitted:.4f}, resid:{residual:.4f}")
                    
                    # Adicionar itens à tabela com ordenação numérica adequada
                    genotype_item = QTableWidgetItem(genotype)
                    self.residuals_values_table.setItem(i, 0, genotype_item)
                    
                    observed_item = QTableWidgetItem(f"{observed:.4f}")
                    observed_item.setData(Qt.UserRole, observed)  # Valor numérico para ordenação
                    self.residuals_values_table.setItem(i, 1, observed_item)
                    
                    fitted_item = QTableWidgetItem(f"{fitted:.4f}")
                    fitted_item.setData(Qt.UserRole, fitted)  # Valor numérico para ordenação
                    self.residuals_values_table.setItem(i, 2, fitted_item)
                    
                    residual_item = QTableWidgetItem(f"{residual:.4f}")
                    residual_item.setData(Qt.UserRole, residual)  # Valor numérico para ordenação
                    self.residuals_values_table.setItem(i, 3, residual_item)
                    
                    residual_pct_item = QTableWidgetItem(f"{residual_pct:.2f}%")
                    residual_pct_item.setData(Qt.UserRole, residual_pct)  # Valor numérico para ordenação
                    self.residuals_values_table.setItem(i, 4, residual_pct_item)
                    
                    residual_abs_item = QTableWidgetItem(f"{residual_abs:.4f}")
                    residual_abs_item.setData(Qt.UserRole, residual_abs)  # Valor numérico para ordenação
                    self.residuals_values_table.setItem(i, 5, residual_abs_item)
                    
                    successful_rows += 1
                    
                    # Colorir linha baseado no tamanho do resíduo (usando std_residuals calculado acima)
                    if residual_abs > std_residuals * 2:  # Outliers (> 2 desvios padrão)
                        color = QColor(255, 200, 200)  # Vermelho claro
                    elif residual_abs > std_residuals:    # Resíduos altos (> 1 desvio padrão)
                        color = QColor(255, 235, 200)  # Laranja claro
                    else:
                        color = QColor(240, 255, 240)  # Verde muito claro
                    
                    # Aplicar cor de fundo à linha
                    for j in range(6):
                        item = self.residuals_values_table.item(i, j)
                        if item:
                            item.setBackground(color)
                            
                except Exception as row_error:
                    print(f"Erro na linha {i}: {row_error}")
                    # Adicionar linha de erro
                    self.residuals_values_table.setItem(i, 0, QTableWidgetItem(f"Erro linha {i}"))
                    for j in range(1, 6):
                        self.residuals_values_table.setItem(i, j, QTableWidgetItem("N/A"))
            
            # Ajustar largura das colunas
            self.residuals_values_table.resizeColumnsToContents()
            print(f"DEBUG: {successful_rows} de {n_rows} linhas processadas com sucesso")  # Debug
            
            # Manter ordem original do arquivo (não aplicar ordenação automática)
            # Usuário pode ordenar manualmente clicando nos cabeçalhos das colunas se desejar
            print("Tabela mantida na ordem original do arquivo")  # Debug
            
        except Exception as e:
            print(f"Erro ao criar tabela de valores detalhados: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: mostrar mensagem de erro na tabela
            self.residuals_values_table.setRowCount(1)
            self.residuals_values_table.setColumnCount(1)
            self.residuals_values_table.setHorizontalHeaderLabels(['Erro'])
            error_item = QTableWidgetItem(f"Erro ao carregar dados: {str(e)}")
            error_item.setBackground(QColor(255, 200, 200))
            self.residuals_values_table.setItem(0, 0, error_item)

    def setup_table_copy_functionality(self, table, table_name=""):
        """Configura funcionalidade de cópia para uma tabela."""
        # Menu de contexto
        table.setContextMenuPolicy(Qt.CustomContextMenu)
        table.customContextMenuRequested.connect(
            lambda pos: self.show_result_table_context_menu(pos, table)
        )
        
        # Atalho Ctrl+C
        copy_shortcut = QShortcut(QKeySequence.Copy, table)
        copy_shortcut.activated.connect(lambda: self.copy_table_selection(table))

    def open_settings(self):
        """Opens the settings dialog."""
        dialog = SettingsDialog(self, {**self.classification_limits, "language_code": self.language_code})
        if dialog.exec_():
            self.classification_limits = dialog.get_settings()
            self.save_settings()
            self.status_bar.showMessage("Configurações salvas.", 3000)
            # Recalculate if data exists
            if not self.indices.empty:
                self.status_bar.showMessage("Configurações salvas. Recalculando com novos limites...", 5000)
                self.start_calculation()

    def load_settings(self):
        """Loads settings from QSettings."""
        self.classification_limits = self.settings_handler.value("classification_limits", {'tolerant': 20, 'mod_tolerant': 40, 'mod_susceptible': 60}, type=dict)

    def save_settings(self):
        """Saves settings to QSettings."""
        self.settings_handler.setValue("classification_limits", self.classification_limits)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accepts drag events if they contain URLs (files)."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Handles dropped files."""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if any(file_path.lower().endswith(ext) for ext in ['.csv', '.txt', '.xlsx', '.xls']):
                self.import_data(file_path)
            else:
                QMessageBox.warning(self, "Formato Inválido", "Por favor, solte apenas arquivos .csv, .txt ou .xlsx.")

    def add_table_row(self):
        """Adds a new row to the input table."""
        current_row_count = self.input_table.rowCount()
        self.input_table.insertRow(current_row_count)
        
        # Add default items to the new row
        for col in range(self.input_table.columnCount()):
            item = QTableWidgetItem("")
            self.input_table.setItem(current_row_count, col, item)
        
        self.status_bar.showMessage(f"Linha {current_row_count + 1} adicionada.", 2000)
    
    def remove_table_row(self):
        """Removes the selected row from the input table."""
        selected_rows = set()
        for item in self.input_table.selectedItems():
            selected_rows.add(item.row())
        
        if not selected_rows:
            current_row = self.input_table.currentRow()
            if current_row >= 0:
                selected_rows.add(current_row)
        
        if not selected_rows:
            QMessageBox.information(self, "Nenhuma Linha Selecionada", 
                                  "Por favor, selecione uma linha para remover.")
            return
        
        # Remove rows in reverse order to maintain correct indices
        for row in sorted(selected_rows, reverse=True):
            self.input_table.removeRow(row)
        
        self.status_bar.showMessage(f"{len(selected_rows)} linha(s) removida(s).", 2000)
    
    def clear_selected_cells(self):
        """Clears the content of selected cells."""
        selected_items = self.input_table.selectedItems()
        
        if not selected_items:
            current_item = self.input_table.currentItem()
            if current_item:
                selected_items = [current_item]
        
        if not selected_items:
            QMessageBox.information(self, "Nenhuma Célula Selecionada", 
                                  "Por favor, selecione as células que deseja limpar.")
            return
        
        for item in selected_items:
            item.setText("")
        
        self.status_bar.showMessage(f"{len(selected_items)} célula(s) limpa(s).", 2000)
    
    def show_table_context_menu(self, position):
        """Shows a context menu for table operations."""
        if self.input_table.itemAt(position) is None:
            return
        
        menu = QMenu(self)
        
        # Add actions
        add_row_action = menu.addAction(self.style().standardIcon(QStyle.SP_ArrowUp), 
                                       "Adicionar Linha")
        add_row_action.triggered.connect(self.add_table_row)
        
        remove_row_action = menu.addAction(self.style().standardIcon(QStyle.SP_ArrowDown), 
                                          "Remover Linha")
        remove_row_action.triggered.connect(self.remove_table_row)
        
        menu.addSeparator()
        
        clear_cells_action = menu.addAction(self.style().standardIcon(QStyle.SP_DialogResetButton), 
                                          "Limpar Células")
        clear_cells_action.triggered.connect(self.clear_selected_cells)
        
        copy_action = menu.addAction(self.style().standardIcon(QStyle.SP_FileIcon), 
                                   "Copiar")
        copy_action.triggered.connect(self.copy_selected_cells)
        
        paste_action = menu.addAction(self.style().standardIcon(QStyle.SP_FileDialogNewFolder), 
                                    "Colar")
        paste_action.triggered.connect(self.paste_cells)
        
        # Show the menu
        menu.exec_(self.input_table.mapToGlobal(position))

    def copy_selected_cells(self):
        """Copies selected cells to clipboard."""
        selected_items = self.input_table.selectedItems()
        
        if not selected_items:
            current_item = self.input_table.currentItem()
            if current_item:
                selected_items = [current_item]
        
        if not selected_items:
            QMessageBox.information(self, "Nenhuma Célula Selecionada", 
                                  "Por favor, selecione as células que deseja copiar.")
            return
        
        # Organize items by row and column
        items_dict = {}
        for item in selected_items:
            row, col = item.row(), item.column()
            if row not in items_dict:
                items_dict[row] = {}
            items_dict[row][col] = item.text()
        
        # Create tab-separated text
        clipboard_text = ""
        sorted_rows = sorted(items_dict.keys())
        for i, row in enumerate(sorted_rows):
            if i > 0:
                clipboard_text += "\n"
            sorted_cols = sorted(items_dict[row].keys())
            for j, col in enumerate(sorted_cols):
                if j > 0:
                    clipboard_text += "\t"
                clipboard_text += items_dict[row][col]
        
        QApplication.clipboard().setText(clipboard_text)
        self.status_bar.showMessage("Células copiadas para a área de transferência.", 2000)
    
    def paste_cells(self):
        self.status_bar.showMessage("Células copiadas para a área de transferência.", 2000)
    
    def paste_cells(self):
        """Pastes cells from clipboard."""
        clipboard_text = QApplication.clipboard().text()
        if not clipboard_text:
            return
        
        current_item = self.input_table.currentItem()
        if not current_item:
            return
        
        start_row = current_item.row()
        start_col = current_item.column()
        
        lines = clipboard_text.split('\n')
        for row_offset, line in enumerate(lines):
            if not line.strip():
                continue
            cells = line.split('\t')
            for col_offset, cell_value in enumerate(cells):
                target_row = start_row + row_offset
                target_col = start_col + col_offset
                
                # Ensure table is large enough
                if target_row >= self.input_table.rowCount():
                    self.input_table.setRowCount(target_row + 1)
                    # Add items to new rows
                    for col in range(self.input_table.columnCount()):
                        if self.input_table.item(target_row, col) is None:
                            self.input_table.setItem(target_row, col, QTableWidgetItem(""))
                
                if target_col < self.input_table.columnCount():
                    item = self.input_table.item(target_row, target_col)
                    if item is None:
                        item = QTableWidgetItem(cell_value)
                        self.input_table.setItem(target_row, target_col, item)
                    else:
                        item.setText(cell_value)
        
        self.status_bar.showMessage("Células coladas da área de transferência.", 2000)

    def save_settings(self):
        """Saves settings to QSettings."""
        try:
            if self.settings_handler:
                self.settings_handler.setValue("classification_limits", self.classification_limits)
        except Exception as e:
            print(f"Warning: Could not save settings: {e}")

    def on_graph_click(self, event):
        """Detecta duplo clique no gráfico para abrir editor interativo."""
        import time
        current_time = time.time()
        
        if current_time - self.last_click_time < 0.5:  # Duplo clique em 0.5s
            self.open_graph_editor(event)
        
        self.last_click_time = current_time
    
    def open_graph_editor(self, event):
        """Abre editor interativo do gráfico."""
        if not self.current_graph_fig:
            return
            
        try:
            # Criar janela de edição
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
            
            editor_window = QDialog(self)
            editor_window.setWindowTitle("Editor de Gráfico - Duplo clique para editar elementos")
            editor_window.setModal(False)
            editor_window.resize(800, 600)
            
            layout = QVBoxLayout(editor_window)
            
            # Canvas editável
            edit_canvas = FigureCanvas(self.current_graph_fig)
            toolbar = NavigationToolbar2QT(edit_canvas, editor_window)
            
            layout.addWidget(toolbar)
            layout.addWidget(edit_canvas)
            
            # Botões
            btn_layout = QHBoxLayout()
            
            apply_btn = QPushButton("Aplicar Alterações")
            apply_btn.clicked.connect(lambda: self.apply_graph_changes(edit_canvas))
            
            close_btn = QPushButton("Fechar")
            close_btn.clicked.connect(editor_window.close)
            
            btn_layout.addWidget(apply_btn)
            btn_layout.addWidget(close_btn)
            layout.addLayout(btn_layout)
            
            # Habilitar edição interativa
            self.current_graph_fig.canvas.mpl_connect('pick_event', self.on_pick)
            
            editor_window.show()
            
        except Exception as e:
            QMessageBox.information(self, "Editor de Gráfico", 
                                  f"Editor interativo disponível!\n\n"
                                  f"Dicas:\n"
                                  f"• Use a barra de ferramentas para zoom/pan\n"
                                  f"• Clique com botão direito nos elementos\n"
                                  f"• Use Ctrl+Z para desfazer\n\n"
                                  f"Erro: {str(e)}")
    
    def on_pick(self, event):
        """Manipula seleção de elementos do gráfico."""
        try:
            artist = event.artist
            if hasattr(artist, 'get_text'):
                # Elemento de texto selecionado
                current_text = artist.get_text()
                new_text, ok = QInputDialog.getText(self, "Editar Texto", 
                                                  "Novo texto:", text=current_text)
                if ok:
                    artist.set_text(new_text)
                    self.current_canvas.draw()
        except Exception as e:
            pass
    
    def apply_graph_changes(self, edit_canvas):
        """Aplica alterações do editor ao gráfico principal."""
        try:
            self.current_canvas.draw()
            self.status_bar.showMessage("Alterações aplicadas ao gráfico!", 3000)
        except Exception as e:
            QMessageBox.warning(self, "Erro", f"Erro ao aplicar alterações: {str(e)}")

    # Report generation methods
    def generate_indices_report(self):
        """Generate and display indices report."""
        try:
            report_content = self.report_generator.generate_indices_report()
            self.show_report_dialog(report_content, "Relatório de Índices de Tolerância")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao gerar relatório de índices: {str(e)}")
    
    def generate_statistics_report(self):
        """Generate and display statistics report."""
        try:
            report_content = self.report_generator.generate_statistics_report()
            self.show_report_dialog(report_content, "Relatório de Análise Estatística")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao gerar relatório estatístico: {str(e)}")
    
    def generate_clustering_report(self):
        """Generate and display clustering report."""
        try:
            # For now, we'll pass None as clustering_result since we need to implement clustering results storage
            report_content = self.report_generator.generate_clustering_report(None)
            self.show_report_dialog(report_content, "Relatório de Análise de Clustering")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao gerar relatório de clustering: {str(e)}")
    
    def show_report_dialog(self, report_content, title):
        """Show report in a dialog with save option."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setGeometry(200, 200, 900, 700)
        
        layout = QVBoxLayout(dialog)
        
        # Text area for report content
        text_edit = QTextEdit()
        text_edit.setPlainText(report_content)
        text_edit.setFont(QFont("Consolas", 10))  # Monospace font for better table formatting
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        save_button = QPushButton("Salvar como...")
        save_button.clicked.connect(lambda: self.save_report(report_content, title))
        button_layout.addWidget(save_button)
        
        close_button = QPushButton("Fechar")
        close_button.clicked.connect(dialog.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        dialog.exec_()
    
    def save_report(self, report_content, title):
        """Save report to file."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            f"Salvar {title}",
            f"{title.replace(' ', '_').lower()}.txt",
            "Arquivos de Texto (*.txt);;Todos os arquivos (*)"
        )
        
        if filename:
            try:
                success = self.report_generator.save_report_to_file(report_content, filename)
                if success is True:
                    QMessageBox.information(self, "Sucesso", f"Relatório salvo em: {filename}")
                    self.status_bar.showMessage(f"Relatório salvo: {filename}", 5000)
                elif isinstance(success, tuple):
                    QMessageBox.critical(self, "Erro", f"Erro ao salvar relatório: {success[1]}")
                else:
                    QMessageBox.critical(self, "Erro", "Erro desconhecido ao salvar relatório")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao salvar relatório: {str(e)}")

    def closeEvent(self, event):
        """Saves settings on application close."""
        self.save_settings()
        super().closeEvent(event)



# --- Application Entry Point ---

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Configure matplotlib backend after QApplication is created
    matplotlib.use('Qt5Agg')
    
    app.setApplicationName("iPASTIC Desktop")
    app.setOrganizationName("CropQ - Agronomia Quantitativa")
    app.setStyle(QStyleFactory.create('Fusion')) # A modern look
    # Set a pleasant default font - increased by 30% for better readability
    font = QFont("Segoe UI", 12)
    app.setFont(font)

    window = MainWindow()
    window.showMaximized()  # Iniciar sempre maximizada
    sys.exit(app.exec_())