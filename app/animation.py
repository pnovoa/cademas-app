import streamlit.components.v1 as components


def render_animated_header():
    """
    Render the CADEMAS-ML pipeline animation in a top-to-bottom flow:
    four inputs → processing → Ri & Ci → prioritization (λ) → analysis views.
    """
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@100..900&display=swap');
        </style>
        <style>
            body {
                margin: 0;
                padding: 0;
                background-color: transparent;
                font-family: 'Geist Mono', monospace;
                overflow: hidden;
            }
            .container {
                width: 100%;
                height: 500px;
                background: linear-gradient(180deg, #0e1117 0%, #151821 55%, #0e1117 100%);
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            svg { width: 100%; height: 100%; max-width: 920px; }

            .node-rect {
                fill: #1f2937;
                stroke: #374151;
                stroke-width: 2px;
                rx: 6px;
            }
            .text-title {
                fill: #f3f4f6;
                font-size: 11px;
                font-weight: 600;
                font-family: 'Geist Mono', monospace;
                pointer-events: none;
            }
            .text-sub {
                fill: #9ca3af;
                font-size: 9px;
                font-family: 'Geist Mono', monospace;
                pointer-events: none;
            }
            .stroke-input { stroke: #6b7280; }
            .stroke-ml { stroke: #3b82f6; }
            .stroke-fuzzy { stroke: #f97316; }
            .stroke-hybrid { stroke: #8b5cf6; }
            .stroke-output { stroke: #10b981; }
            .stroke-process { stroke: #64748b; }

            .path-line {
                fill: none;
                stroke: #4b5563;
                stroke-width: 2px;
                opacity: 0.35;
            }
            .dot { filter: drop-shadow(0 0 4px rgba(255,255,255,0.7)); }
            .dot-input { fill: #d1d5db; }
            .dot-ml { fill: #60a5fa; }
            .dot-fuzzy { fill: #fb923c; }
            .dot-hybrid { fill: #a78bfa; }
            .dot-output { fill: #34d399; }
        </style>
    </head>
    <body>
        <div class="container">
            <svg viewBox="0 0 720 470" preserveAspectRatio="xMidYMid meet">
                <defs>
                    <!-- Inputs → Processing -->
                    <path id="in1" d="M 100 62 L 100 108 L 360 108 L 360 128" />
                    <path id="in2" d="M 260 62 L 260 108 L 360 108" />
                    <path id="in3" d="M 420 62 L 420 108 L 360 108" />
                    <path id="in4" d="M 580 62 L 580 108 L 360 108" />
                    <!-- Processing → Ri / Ci -->
                    <path id="proc-ri" d="M 330 172 L 280 172 L 280 198" />
                    <path id="proc-ci" d="M 390 172 L 440 172 L 440 198" />
                    <!-- Ri / Ci → Prioritization -->
                    <path id="ri-prior" d="M 280 242 L 280 268 L 360 268 L 360 288" />
                    <path id="ci-prior" d="M 440 242 L 440 268 L 360 268" />
                    <!-- Prioritization → Outputs -->
                    <path id="prior-out1" d="M 300 332 L 300 358 L 105 358 L 105 378" />
                    <path id="prior-out2" d="M 340 332 L 340 358 L 265 358 L 265 378" />
                    <path id="prior-out3" d="M 380 332 L 380 358 L 425 358 L 425 378" />
                    <path id="prior-out4" d="M 420 332 L 420 358 L 585 358 L 585 378" />
                </defs>

                <!-- Connector lines -->
                <path d="M 100 62 L 100 108 L 360 108 L 360 128" class="path-line" />
                <path d="M 260 62 L 260 108 L 360 108" class="path-line" />
                <path d="M 420 62 L 420 108 L 360 108" class="path-line" />
                <path d="M 580 62 L 580 108 L 360 108" class="path-line" />
                <path d="M 330 172 L 280 172 L 280 198" class="path-line" />
                <path d="M 390 172 L 440 172 L 440 198" class="path-line" />
                <path d="M 280 242 L 280 268 L 360 268 L 360 288" class="path-line" />
                <path d="M 440 242 L 440 268 L 360 268" class="path-line" />
                <path d="M 300 332 L 300 358 L 105 358 L 105 378" class="path-line" />
                <path d="M 340 332 L 340 358 L 265 358 L 265 378" class="path-line" />
                <path d="M 380 332 L 380 358 L 425 358 L 425 378" class="path-line" />
                <path d="M 420 332 L 420 358 L 585 358 L 585 378" class="path-line" />

                <!-- Level 1: four inputs -->
                <g transform="translate(25, 20)">
                    <rect width="150" height="42" class="node-rect stroke-input" />
                    <text x="75" y="18" text-anchor="middle" class="text-title">Model config</text>
                    <text x="75" y="31" text-anchor="middle" class="text-sub">JSON</text>
                </g>
                <g transform="translate(185, 20)">
                    <rect width="150" height="42" class="node-rect stroke-input" />
                    <text x="75" y="18" text-anchor="middle" class="text-title">Context config</text>
                    <text x="75" y="31" text-anchor="middle" class="text-sub">JSON</text>
                </g>
                <g transform="translate(345, 20)">
                    <rect width="150" height="42" class="node-rect stroke-input" />
                    <text x="75" y="18" text-anchor="middle" class="text-title">MOJO models</text>
                    <text x="75" y="31" text-anchor="middle" class="text-sub">.zip</text>
                </g>
                <g transform="translate(505, 20)">
                    <rect width="150" height="42" class="node-rect stroke-input" />
                    <text x="75" y="18" text-anchor="middle" class="text-title">Case dataset</text>
                    <text x="75" y="31" text-anchor="middle" class="text-sub">CSV</text>
                </g>

                <!-- Level 2: processing -->
                <g transform="translate(265, 128)">
                    <rect width="190" height="44" class="node-rect stroke-process" />
                    <text x="95" y="19" text-anchor="middle" class="text-title">Analysis pipeline</text>
                    <text x="95" y="32" text-anchor="middle" class="text-sub">ML inference + fuzzy context</text>
                </g>

                <!-- Level 3: Ri & Ci -->
                <g transform="translate(200, 198)">
                    <rect width="160" height="44" class="node-rect stroke-ml" />
                    <text x="80" y="19" text-anchor="middle" class="text-title">Global ML Risk</text>
                    <text x="80" y="32" text-anchor="middle" class="text-sub">Ri</text>
                </g>
                <g transform="translate(360, 198)">
                    <rect width="160" height="44" class="node-rect stroke-fuzzy" />
                    <text x="80" y="19" text-anchor="middle" class="text-title">Context Alignment</text>
                    <text x="80" y="32" text-anchor="middle" class="text-sub">Ci</text>
                </g>

                <!-- Level 4: prioritization -->
                <g transform="translate(235, 288)">
                    <rect width="250" height="44" class="node-rect stroke-hybrid" />
                    <text x="125" y="19" text-anchor="middle" class="text-title">Prioritization Score</text>
                    <text x="125" y="32" text-anchor="middle" class="text-sub">λ · Ri + (1 − λ) · Ci</text>
                </g>

                <!-- Level 5: analysis views -->
                <g transform="translate(30, 378)">
                    <rect width="150" height="42" class="node-rect stroke-output" />
                    <text x="75" y="18" text-anchor="middle" class="text-title">Overview</text>
                    <text x="75" y="31" text-anchor="middle" class="text-sub">prioritized cases</text>
                </g>
                <g transform="translate(190, 378)">
                    <rect width="150" height="42" class="node-rect stroke-output" />
                    <text x="75" y="18" text-anchor="middle" class="text-title">Models</text>
                    <text x="75" y="31" text-anchor="middle" class="text-sub">weights &amp; risks</text>
                </g>
                <g transform="translate(350, 378)">
                    <rect width="150" height="42" class="node-rect stroke-output" />
                    <text x="75" y="18" text-anchor="middle" class="text-title">Context</text>
                    <text x="75" y="31" text-anchor="middle" class="text-sub">rules &amp; membership</text>
                </g>
                <g transform="translate(510, 378)">
                    <rect width="150" height="42" class="node-rect stroke-output" />
                    <text x="75" y="18" text-anchor="middle" class="text-title">Robustness</text>
                    <text x="75" y="31" text-anchor="middle" class="text-sub">λ sensitivity</text>
                </g>

                <!-- Animated particles: inputs → processing -->
                <circle r="4" class="dot dot-input">
                    <animateMotion dur="2.2s" repeatCount="indefinite" calcMode="linear">
                        <mpath href="#in1"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;1;1;0" dur="2.2s" repeatCount="indefinite" />
                </circle>
                <circle r="4" class="dot dot-input">
                    <animateMotion dur="2.2s" begin="0.3s" repeatCount="indefinite" calcMode="linear">
                        <mpath href="#in2"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;1;1;0" dur="2.2s" begin="0.3s" repeatCount="indefinite" />
                </circle>
                <circle r="4" class="dot dot-input">
                    <animateMotion dur="2.2s" begin="0.6s" repeatCount="indefinite" calcMode="linear">
                        <mpath href="#in3"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;1;1;0" dur="2.2s" begin="0.6s" repeatCount="indefinite" />
                </circle>
                <circle r="4" class="dot dot-input">
                    <animateMotion dur="2.2s" begin="0.9s" repeatCount="indefinite" calcMode="linear">
                        <mpath href="#in4"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;1;1;0" dur="2.2s" begin="0.9s" repeatCount="indefinite" />
                </circle>

                <!-- Processing → Ri / Ci -->
                <circle r="4" class="dot dot-ml">
                    <animateMotion dur="1.8s" begin="1.2s" repeatCount="indefinite" calcMode="linear">
                        <mpath href="#proc-ri"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;1;1;0" dur="1.8s" begin="1.2s" repeatCount="indefinite" />
                </circle>
                <circle r="4" class="dot dot-fuzzy">
                    <animateMotion dur="1.8s" begin="1.4s" repeatCount="indefinite" calcMode="linear">
                        <mpath href="#proc-ci"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;1;1;0" dur="1.8s" begin="1.4s" repeatCount="indefinite" />
                </circle>

                <!-- Ri / Ci → Prioritization -->
                <circle r="5" class="dot dot-ml">
                    <animateMotion dur="2s" begin="2.2s" repeatCount="indefinite" calcMode="linear">
                        <mpath href="#ri-prior"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;1;1;0" dur="2s" begin="2.2s" repeatCount="indefinite" />
                </circle>
                <circle r="5" class="dot dot-fuzzy">
                    <animateMotion dur="2s" begin="2.4s" repeatCount="indefinite" calcMode="linear">
                        <mpath href="#ci-prior"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;1;1;0" dur="2s" begin="2.4s" repeatCount="indefinite" />
                </circle>

                <!-- Prioritization → outputs -->
                <circle r="4" class="dot dot-hybrid">
                    <animateMotion dur="2.4s" begin="3s" repeatCount="indefinite" calcMode="linear">
                        <mpath href="#prior-out1"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;1;1;0" dur="2.4s" begin="3s" repeatCount="indefinite" />
                </circle>
                <circle r="4" class="dot dot-output">
                    <animateMotion dur="2.4s" begin="3.2s" repeatCount="indefinite" calcMode="linear">
                        <mpath href="#prior-out2"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;1;1;0" dur="2.4s" begin="3.2s" repeatCount="indefinite" />
                </circle>
                <circle r="4" class="dot dot-output">
                    <animateMotion dur="2.4s" begin="3.4s" repeatCount="indefinite" calcMode="linear">
                        <mpath href="#prior-out3"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;1;1;0" dur="2.4s" begin="3.4s" repeatCount="indefinite" />
                </circle>
                <circle r="4" class="dot dot-output">
                    <animateMotion dur="2.4s" begin="3.6s" repeatCount="indefinite" calcMode="linear">
                        <mpath href="#prior-out4"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;1;1;0" dur="2.4s" begin="3.6s" repeatCount="indefinite" />
                </circle>
            </svg>
        </div>
    </body>
    </html>
    """
    components.html(html_code, height=510, scrolling=False)
