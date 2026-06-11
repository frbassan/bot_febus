import re

with open("combined_app.py", "r") as f:
    text = f.read()

# 1. Replace matplotlib imports in both modules
text = text.replace("import matplotlib.pyplot as plt", "import plotly.graph_objects as go")

# 2. Replace render_2d_analysis_module plotting block
old_2d_plot = """                    fig, ax = plt.subplots(figsize=(10, 5))
                    alpha_orig = 0.3 if len(plots_to_make) > 0 else 1.0
                    ax.plot(data_x, data_y, label=f'Raw {y_dataset}', color='black', alpha=alpha_orig, linestyle='-' if alpha_orig == 1.0 else ':')
                    for pt in plots_to_make: ax.plot(data_x, pt[0], label=pt[1], color=pt[2], linewidth=1.5)
            
                    # Matplotlib Configuration
                    ax.set_xlabel(x_dataset, labelpad=15, loc='center') # O loc center e labelpad que colocamos
                    ax.set_ylabel(y_dataset, labelpad=20, loc='center')
                    ax.set_title(f"2D Signal Profile: {section_title} ({y_dataset} vs {x_dataset})", pad=20)
                    if ymin is not None and ymax is not None:
                        ax.set_ylim([ymin, ymax])
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend()
                    st.pyplot(fig)"""

new_2d_plot = """                    # Downsample for Plotly to maintain performance
                    MAX_POINTS = 1500
                    if len(data_x) > MAX_POINTS:
                        step = len(data_x) // MAX_POINTS
                        plot_x = data_x[::step]
                        plot_y = data_y[::step]
                        downsample_msg = f" (Downsampled for plot: {len(plot_x)} pts)"
                    else:
                        step = 1
                        plot_x = data_x
                        plot_y = data_y
                        downsample_msg = ""

                    fig = go.Figure()
                    alpha_orig = 0.4 if len(plots_to_make) > 0 else 1.0
                    fig.add_trace(go.Scatter(
                        x=plot_x, y=plot_y, mode='lines',
                        name=f'Raw {y_dataset}',
                        line=dict(color=f'rgba(0, 0, 0, {alpha_orig})', dash='dot' if alpha_orig < 1.0 else 'solid')
                    ))
                    
                    color_map = {'tab:orange': 'orange', 'tab:green': 'green', 'tab:purple': 'purple', 'tab:red': 'red', 'tab:cyan': 'cyan'}
                    for pt in plots_to_make:
                        p_data = pt[0][::step] if step > 1 else pt[0]
                        c_str = color_map.get(pt[2], 'blue')
                        fig.add_trace(go.Scatter(
                            x=plot_x, y=p_data, mode='lines',
                            name=f'Filter: {pt[1]}',
                            line=dict(color=c_str, width=2)
                        ))
                        
                    fig.update_layout(
                        title=f"2D Signal Profile: {section_title} {downsample_msg}",
                        xaxis_title=x_dataset, yaxis_title=y_dataset,
                        hovermode="x unified", template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    if ymin is not None and ymax is not None:
                        fig.update_yaxes(range=[ymin, ymax])
                        
                    st.plotly_chart(fig, use_container_width=True)"""

text = text.replace(old_2d_plot, new_2d_plot)


# 3. Replace render_time_series_module plotting block
old_ts_plot = """            fig, ax = plt.subplots(figsize=(10, 4))
            
            MAX_POINTS = 1500
            if len(data_x) > MAX_POINTS:
                step = len(data_x) // MAX_POINTS
                plot_x = data_x[::step]
                plot_y = data_y[::step]
                downsample_msg = f" (Downsampled for plot: {len(plot_x)} pts)"
            else:
                plot_x = data_x
                plot_y = data_y
                downsample_msg = ""
                
            ax.plot(plot_x, plot_y, label='Raw Time-Series', color='tab:blue', alpha=0.6 if plots_to_make else 1.0, linewidth=1.5)
            
            for p_data, p_name, p_color in plots_to_make:
                if len(p_data) > MAX_POINTS:
                    p_data = p_data[::step]
                ax.plot(plot_x, p_data, label=f'Filter: {p_name}', color=p_color, linewidth=1.5)
                
            ax.set_title(f"{section_title}{downsample_msg}", fontweight='bold')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_dataset)
            if ymin is not None and ymax is not None: ax.set_ylim([ymin, ymax])
            
            import matplotlib.dates as mdates
            if len(plot_x) > 0 and isinstance(plot_x[0], datetime):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y %H:%M'))
                fig.autofmt_xdate(rotation=45)
                
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='best')
            fig.tight_layout()
            st.pyplot(fig)"""

new_ts_plot = """            MAX_POINTS = 1500
            if len(data_x) > MAX_POINTS:
                step = len(data_x) // MAX_POINTS
                plot_x = data_x[::step]
                plot_y = data_y[::step]
                downsample_msg = f" (Downsampled for plot: {len(plot_x)} pts)"
            else:
                step = 1
                plot_x = data_x
                plot_y = data_y
                downsample_msg = ""
                
            fig = go.Figure()
            alpha_orig = 0.4 if len(plots_to_make) > 0 else 1.0
            fig.add_trace(go.Scatter(
                x=plot_x, y=plot_y, mode='lines',
                name='Raw Time-Series',
                line=dict(color=f'rgba(0, 100, 255, {alpha_orig})')
            ))
            
            color_map = {'tab:orange': 'orange', 'tab:green': 'green', 'tab:purple': 'purple', 'tab:red': 'red', 'tab:cyan': 'cyan'}
            for pt in plots_to_make:
                p_data = pt[0][::step] if step > 1 else pt[0]
                c_str = color_map.get(pt[2], 'blue')
                fig.add_trace(go.Scatter(
                    x=plot_x, y=p_data, mode='lines',
                    name=f'Filter: {pt[1]}',
                    line=dict(color=c_str, width=2)
                ))
                
            fig.update_layout(
                title=f"{section_title}{downsample_msg}",
                xaxis_title=x_label, yaxis_title=y_dataset,
                hovermode="x unified", template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            if len(plot_x) > 0 and isinstance(plot_x[0], datetime):
                fig.update_xaxes(tickformat="%d/%m/%y %H:%M")
                
            if ymin is not None and ymax is not None:
                fig.update_yaxes(range=[ymin, ymax])
                
            st.plotly_chart(fig, use_container_width=True)"""

text = text.replace(old_ts_plot, new_ts_plot)

with open("combined_app.py", "w") as f:
    f.write(text)

print("Applied Plotly Migration!")
