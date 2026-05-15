"""
Анализ культурных маркеров и профиля культурной плотности
"""

import re
import numpy as np
import plotly.graph_objects as go

class CulturalAnalyzer:
    def __init__(self, cultural_markers):
        self.markers = cultural_markers

    def find_markers_in_text(self, text):
        text_lower = text.lower()
        found_markers = []
        seen = set()
        
        for marker in self.markers:
            if marker in seen:
                continue
                
            pattern = r'\b' + re.escape(marker) + r'\b'
            if re.search(pattern, text_lower):
                seen.add(marker)
                found_markers.append({
                    'marker': marker,
                    'context': self._get_context(text, text_lower.find(marker), len(marker))
                })
        
        return found_markers
    
    def find_marker_positions(self, text):
        text_lower = text.lower()
        found = []
        
        for marker in self.markers:
            pattern = r'\b' + re.escape(marker) + r'\b'
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                start = max(0, match.start() - 80)
                end = min(len(text), match.end() + 80)
                context = text[start:end]
                if start > 0:
                    context = "..." + context
                if end < len(text):
                    context = context + "..."
                    
                found.append({
                    'marker': marker,
                    'start': match.start(),
                    'end': match.end(),
                    'context': context
                })
        
        return found

    def find_marker_sentences(self, text, sentences):
        markers = self.find_markers_in_text(text)
        result = []

        for marker in markers:
            for sent in sentences:
                if marker['marker'] in sent.lower():
                    result.append({
                        'marker': marker['marker'],
                        'sentence': sent,
                        'context': marker['context']
                    })
                    break

        return result
    
    def _get_context(self, text, pos, length, context_chars=80):
        start = max(0, pos - context_chars)
        end = min(len(text), pos + length + context_chars)
        context = text[start:end]
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
        return context


class CulturalDensityAnalyzer:
    def __init__(self, cultural_analyzer):
        self.cultural_analyzer = cultural_analyzer

    def calculate_profile(self, text, window_size=100):
        markers = self.cultural_analyzer.find_marker_positions(text)

        if not markers:
            return {
                'density_profile': [],
                'total_markers': 0,
                'max_density': 0,
                'avg_density': 0,
                'peaks': []
            }

        words = text.split()
        total_words = len(words)

        marker_positions = []
        for marker in markers:
            word_index = len(text[:marker['start']].split())
            marker_positions.append(word_index)

        density_profile = []
        for i in range(0, total_words, window_size // 2):
            window_end = min(i + window_size, total_words)
            marker_count = sum(1 for pos in marker_positions if i <= pos < window_end)
            density = marker_count / window_size * 100
            density_profile.append({
                'start': i,
                'end': window_end,
                'density': round(density, 2),
                'count': marker_count
            })

        densities = [d['density'] for d in density_profile]
        avg_density = np.mean(densities) if densities else 0

        peaks = []
        for i in range(1, len(density_profile) - 1):
            if (density_profile[i]['density'] > density_profile[i - 1]['density'] and
                density_profile[i]['density'] > density_profile[i + 1]['density'] and
                density_profile[i]['density'] > avg_density * 1.5):
                peaks.append(density_profile[i])

        return {
            'density_profile': density_profile,
            'total_markers': len(set([m['marker'] for m in markers])),
            'max_density': max(densities) if densities else 0,
            'avg_density': round(avg_density, 2),
            'peaks': peaks
        }

    def create_plot(self, profile):
        if not profile['density_profile']:
            return "<div>Культурные маркеры не обнаружены</div>"

        x = [f"{d['start']}-{d['end']}" for d in profile['density_profile']]
        y = [d['density'] for d in profile['density_profile']]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines+markers',
            name='Плотность маркеров',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8, color='#764ba2'),
            hovertemplate='Позиция: %{x}<br>Плотность: %{y:.1f}<extra></extra>'
        ))

        fig.add_hline(y=profile['avg_density'], line_dash="dash", line_color="orange",
                      annotation_text=f"Среднее: {profile['avg_density']}")

        if profile['peaks']:
            peak_x = [f"{p['start']}-{p['end']}" for p in profile['peaks']]
            peak_y = [p['density'] for p in profile['peaks']]
            fig.add_trace(go.Scatter(
                x=peak_x, y=peak_y, mode='markers',
                name='Пики плотности',
                marker=dict(symbol='star', size=15, color='red')
            ))

        fig.update_layout(
            height=450,
            xaxis_title="Позиция в тексте (слова)",
            yaxis_title="Плотность (маркеров на 100 слов)",
            hovermode='closest',
            plot_bgcolor='rgba(240,240,240,0.5)'
        )

        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text=f"Всего уникальных маркеров: {profile['total_markers']} | Пиковая плотность: {profile['max_density']}",
            showarrow=False,
            bgcolor='white',
            bordercolor='#667eea',
            borderwidth=1
        )

        return fig.to_html(full_html=False, include_plotlyjs='cdn')