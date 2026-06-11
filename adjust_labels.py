import re

with open("combined_app.py", "r") as f:
    text = f.read()

# 1. 2D Spatial module line color
text = text.replace(
    "line=dict(color=f'rgba(0, 0, 0, {alpha_orig})', dash='dot' if alpha_orig < 1.0 else 'solid')",
    "line=dict(color=f'rgba(0, 100, 255, {alpha_orig})', dash='dot' if alpha_orig < 1.0 else 'solid')"
)
text = text.replace(
    "name=f'Raw {y_dataset}',",
    "name=f'Raw Data',"
)

# 2. 2D Spatial layout
old_layout_2d = """                    fig.update_layout(
                        title=f"2D Signal Profile: {section_title} {downsample_msg}",
                        xaxis_title=x_dataset, yaxis_title=y_dataset,
                        hovermode="x unified", template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    if ymin is not None and ymax is not None:
                        fig.update_yaxes(range=[ymin, ymax])"""

new_layout_2d = """                    label_map = {"temp_data": "Temperature (°C)", "strain_data": "Strain (uE)"}
                    display_y = label_map.get(y_dataset, y_dataset)
                    fig.update_layout(
                        title=f"2D Signal Profile: {section_title} {downsample_msg}",
                        xaxis_title=x_dataset, yaxis_title=display_y,
                        hovermode="x unified", template="plotly_white",
                        plot_bgcolor="#f8f9fa", paper_bgcolor="white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    fig.update_xaxes(showgrid=True, gridcolor='white', gridwidth=1.5, zeroline=True, zerolinecolor='lightgrey')
                    fig.update_yaxes(showgrid=True, gridcolor='white', gridwidth=1.5, zeroline=True, zerolinecolor='lightgrey')
                    
                    if ymin is not None and ymax is not None:
                        fig.update_yaxes(range=[ymin, ymax])"""

text = text.replace(old_layout_2d, new_layout_2d)

# 3. Time-Series module layout
old_layout_ts = """            fig.update_layout(
                title=f"{section_title}{downsample_msg}",
                xaxis_title=x_label, yaxis_title=y_dataset,
                hovermode="x unified", template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            if len(plot_x) > 0 and isinstance(plot_x[0], datetime):
                fig.update_xaxes(tickformat="%d/%m/%y %H:%M")
                
            if ymin is not None and ymax is not None:
                fig.update_yaxes(range=[ymin, ymax])"""

new_layout_ts = """            label_map = {"temp_data": "Temperature (°C)", "strain_data": "Strain (uE)"}
            display_y = label_map.get(y_dataset, y_dataset)
            x_title = "Time (dd/mm/yy hh:mm)" if len(plot_x) > 0 and isinstance(plot_x[0], datetime) else x_label
            fig.update_layout(
                title=f"{section_title}{downsample_msg}",
                xaxis_title=x_title, yaxis_title=display_y,
                hovermode="x unified", template="plotly_white",
                plot_bgcolor="#f8f9fa", paper_bgcolor="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_xaxes(showgrid=True, gridcolor='white', gridwidth=1.5, zeroline=True, zerolinecolor='lightgrey')
            fig.update_yaxes(showgrid=True, gridcolor='white', gridwidth=1.5, zeroline=True, zerolinecolor='lightgrey')
            
            if len(plot_x) > 0 and isinstance(plot_x[0], datetime):
                fig.update_xaxes(tickformat="%d/%m/%y %H:%M")
                
            if ymin is not None and ymax is not None:
                fig.update_yaxes(range=[ymin, ymax])"""

text = text.replace(old_layout_ts, new_layout_ts)

with open("combined_app.py", "w") as f:
    f.write(text)

print("Adjustments applied!")
