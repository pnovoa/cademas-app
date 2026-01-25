import streamlit.components.v1 as components

def render_animated_header():
    """
    Renderiza el diagrama de flujo animado usando un iframe aislado
    para garantizar que los estilos y animaciones funcionen en cualquier navegador.
    """
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
@import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@100..900&display=swap');
</style>
        <style>
            body { margin: 0; padding: 0; background-color: transparent; font-family: 'Geist Mono'; overflow: hidden; }
            .container {
                width: 100%;
                height: 180px;
                background: linear-gradient(90deg, #0e1117 0%, #1a1c24 100%);
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            svg { width: 100%; height: 100%; max-width: 900px; }

            /* Estilos estáticos */
            .node-rect { fill: #1f2937; stroke: #374151; stroke-width: 2px; rx: 6px; }
            .text-title { fill: #f3f4f6; font-size: 12px; font-weight: semibold; font-family: 'Geist Mono', monospace; pointer-events: none; }
            .text-sub { fill: #9ca3af; font-size: 10px; font-family: 'Geist Mono', monospace; pointer-events: none; }

            /* Colores de los nodos */
            .stroke-ml { stroke: #3b82f6; }
            .stroke-fuzzy { stroke: #f97316; }
            .stroke-hybrid { stroke: #8b5cf6; }

            /* Caminos */
            .path-line { fill: none; stroke: #4b5563; stroke-width: 2px; opacity: 0.3; }

            /* Partículas brillantes */
            .dot { fill: white; filter: drop-shadow(0 0 4px rgba(255,255,255,0.8)); }
            .dot-ml { fill: #60a5fa; }
            .dot-fuzzy { fill: #fb923c; }
            .dot-hybrid { fill: #a78bfa; }
        </style>
    </head>
    <body>
        <div class="container">
            <svg viewBox="0 0 800 160" preserveAspectRatio="xMidYMid meet">
                <defs>
                    <path id="p1" d="M 90 50 L 240 50" />
                    <path id="p2" d="M 90 110 L 240 110" />
                    <path id="p3" d="M 340 50 L 460 80" />
                    <path id="p4" d="M 340 110 L 460 80" />
                    <path id="p5" d="M 560 80 L 670 80" />
                </defs>

                <path d="M 90 50 L 240 50" class="path-line" />
                <path d="M 90 110 L 240 110" class="path-line" />
                <path d="M 340 50 L 460 80" class="path-line" />
                <path d="M 340 110 L 460 80" class="path-line" />
                <path d="M 560 80 L 670 80" class="path-line" />

                <g transform="translate(10, 30)">
                    <rect width="80" height="40" class="node-rect" />
                    <text x="40" y="20" text-anchor="middle" dominant-baseline="middle" class="text-title">DATA</text>
                    <text x="40" y="32" text-anchor="middle" class="text-sub">CSV</text>
                </g>
                <g transform="translate(10, 90)">
                    <rect width="80" height="40" class="node-rect" />
                    <text x="40" y="20" text-anchor="middle" dominant-baseline="middle" class="text-title">RULES</text>
                    <text x="40" y="32" text-anchor="middle" class="text-sub">JSON</text>
                </g>

                <g transform="translate(240, 30)">
                    <rect width="100" height="40" class="node-rect stroke-ml" />
                    <text x="50" y="20" text-anchor="middle" dominant-baseline="middle" class="text-title">ML ENSEMBLE</text>
                    <text x="50" y="32" text-anchor="middle" class="text-sub">Riesgo (Ri)</text>
                </g>
                <g transform="translate(240, 90)">
                    <rect width="100" height="40" class="node-rect stroke-fuzzy" />
                    <text x="50" y="20" text-anchor="middle" dominant-baseline="middle" class="text-title">FUZZY LOGIC</text>
                    <text x="50" y="32" text-anchor="middle" class="text-sub">Contexto (Ci)</text>
                </g>

                <g transform="translate(460, 60)">
                    <rect width="100" height="40" class="node-rect stroke-hybrid" />
                    <text x="50" y="20" text-anchor="middle" dominant-baseline="middle" class="text-title">HYBRID CORE</text>
                    <text x="50" y="32" text-anchor="middle" class="text-sub">λ•Ri + (1-λ)•Ci</text>
                </g>

                <g transform="translate(670, 60)">
                    <rect width="100" height="40" class="node-rect" />
                    <text x="50" y="20" text-anchor="middle" dominant-baseline="middle" class="text-title">DASHBOARD</text>
                    <text x="50" y="32" text-anchor="middle" class="text-sub">Decisión</text>
                </g>

                <circle r="4" class="dot dot-ml">
                    <animateMotion dur="2s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" calcMode="linear">
                        <mpath href="#p1"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;1;1;0" dur="2s" repeatCount="indefinite" />
                </circle>

                <circle r="4" class="dot dot-fuzzy">
                    <animateMotion dur="2.5s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" calcMode="linear">
                        <mpath href="#p2"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;1;1;0" dur="2.5s" repeatCount="indefinite" />
                </circle>

                <circle r="4" class="dot dot-ml">
                    <animateMotion dur="2s" begin="1s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" calcMode="linear">
                        <mpath href="#p3"/>
                    </animateMotion>
                     <animate attributeName="opacity" values="0;1;1;0" dur="2s" begin="1s" repeatCount="indefinite" />
                </circle>

                <circle r="4" class="dot dot-fuzzy">
                    <animateMotion dur="2.5s" begin="1.2s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" calcMode="linear">
                        <mpath href="#p4"/>
                    </animateMotion>
                     <animate attributeName="opacity" values="0;1;1;0" dur="2.5s" begin="1.2s" repeatCount="indefinite" />
                </circle>

                 <circle r="5" class="dot dot-hybrid">
                    <animateMotion dur="3s" begin="0.5s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" calcMode="linear">
                        <mpath href="#p5"/>
                    </animateMotion>
                     <animate attributeName="opacity" values="0;1;1;0" dur="3s" begin="0.5s" repeatCount="indefinite" />
                </circle>

            </svg>
        </div>
    </body>
    </html>
    """
    # Renderizamos el componente HTML con una altura fija para evitar scrollbars
    components.html(html_code, height=190, scrolling=False)