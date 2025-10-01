from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List

from ...database import get_db
from ...models import Composite, ECMAttempt, Factor

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    """
    Simple dashboard showing all factorization results.
    """

    # Get recent composites with their attempts and factors
    composites = db.query(Composite).order_by(desc(Composite.created_at)).limit(50).all()

    # Get recent attempts
    attempts = db.query(ECMAttempt).order_by(desc(ECMAttempt.created_at)).limit(100).all()

    # Get all factors
    factors = db.query(Factor).order_by(desc(Factor.created_at)).all()

    # Build summary stats
    total_composites = db.query(Composite).count()
    total_attempts = db.query(ECMAttempt).count()
    total_factors = db.query(Factor).count()
    fully_factored = db.query(Composite).filter(Composite.is_fully_factored == True).count()

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ECM Distributed Factorization Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .stats {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .stat-item {{ display: inline-block; margin-right: 30px; }}
            .search-box {{ margin-bottom: 20px; }}
            .search-box input {{ padding: 10px; width: 100%; max-width: 500px; font-size: 14px; border: 1px solid #ddd; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .number {{ font-family: monospace; word-break: break-all; }}
            .factor {{ color: #d63384; font-weight: bold; }}
            .success {{ color: #198754; }}
            .pending {{ color: #fd7e14; }}
            .section {{ margin-bottom: 40px; }}
            .small-text {{ font-size: 0.9em; color: #666; }}
            .progress {{ background: #e9ecef; border-radius: 10px; height: 20px; overflow: hidden; display: inline-block; width: 100px; vertical-align: middle; }}
            .progress-bar {{ background: #198754; height: 100%; transition: width 0.3s; }}
            .btn {{ background: #007bff; color: white; padding: 5px 10px; border: none; border-radius: 3px; cursor: pointer; text-decoration: none; display: inline-block; font-size: 0.85em; }}
            .btn:hover {{ background: #0056b3; }}
            tr.clickable-row {{ cursor: pointer; }}
            tr.clickable-row:hover {{ background-color: #f8f9fa; }}
        </style>
    </head>
    <body>
        <h1>ECM Distributed Factorization Dashboard</h1>

        <div class="stats">
            <h3>Summary Statistics</h3>
            <div class="stat-item"><strong>Total Composites:</strong> {total_composites}</div>
            <div class="stat-item"><strong>Total Attempts:</strong> {total_attempts}</div>
            <div class="stat-item"><strong>Factors Found:</strong> {total_factors}</div>
            <div class="stat-item"><strong>Fully Factored:</strong> {fully_factored}</div>
        </div>

        <div class="section">
            <h2>Recent Composites</h2>
            <div class="search-box">
                <input type="text" id="searchInput" placeholder="Search by number or composite..." onkeyup="searchTable()">
            </div>
            <table id="compositesTable">
                <thead>
                    <tr>
                        <th>Number</th>
                        <th>Digit Length</th>
                        <th>Status</th>
                        <th>T-Level Progress</th>
                        <th>Attempts</th>
                        <th>Factors</th>
                        <th>Added</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for comp in composites:
        attempt_count = db.query(ECMAttempt).filter(ECMAttempt.composite_id == comp.id).count()
        comp_factors = db.query(Factor).filter(Factor.composite_id == comp.id).all()

        if comp.is_prime:
            status = '<span class="success">Prime</span>'
        elif comp.is_fully_factored:
            status = '<span class="success">Fully Factored</span>'
        else:
            status = '<span class="pending">Composite</span>'

        factors_display = ""
        if comp_factors:
            factors_display = " × ".join([f'<span class="factor">{f.factor}</span>' for f in comp_factors])

        # T-level progress display
        current_t = comp.current_t_level or 0.0
        target_t = comp.target_t_level or 0.0
        t_progress_pct = (current_t / target_t * 100) if target_t > 0 else 0
        t_progress_color = "#198754" if t_progress_pct >= 100 else "#fd7e14" if t_progress_pct >= 50 else "#dc3545"

        t_level_display = ""
        if target_t > 0:
            t_level_display = f'''
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div class="progress">
                        <div class="progress-bar" style="width: {min(t_progress_pct, 100)}%; background-color: {t_progress_color};"></div>
                    </div>
                    <span style="font-size: 0.85em; font-weight: bold; color: {t_progress_color};">
                        t{current_t:.1f} / t{target_t:.1f}
                    </span>
                </div>
            '''
        else:
            t_level_display = '<span style="color: #999;">N/A</span>'

        html_content += f"""
                    <tr>
                        <td class="number">{comp.number[:50]}{'...' if len(comp.number) > 50 else ''}</td>
                        <td>{comp.digit_length}</td>
                        <td>{status}</td>
                        <td>{t_level_display}</td>
                        <td>{attempt_count}</td>
                        <td>{factors_display or 'None'}</td>
                        <td class="small-text">{comp.created_at.strftime('%Y-%m-%d %H:%M')}</td>
                        <td><a href="/api/v1/dashboard/composites/{comp.id}/details" class="btn">Details</a></td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Recent Factorization Attempts</h2>
            <table>
                <thead>
                    <tr>
                        <th>Composite</th>
                        <th>Method</th>
                        <th>B1</th>
                        <th>Curves</th>
                        <th>Factor Found</th>
                        <th>Client</th>
                        <th>Time</th>
                        <th>Submitted</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for attempt in attempts:
        comp = db.query(Composite).filter(Composite.id == attempt.composite_id).first()
        comp_display = comp.number[:30] + '...' if comp and len(comp.number) > 30 else (comp.number if comp else 'Unknown')
        
        factor_display = "None"
        if attempt.factor_found:
            factor_display = f'<span class="factor">{attempt.factor_found}</span>'
        
        html_content += f"""
                    <tr>
                        <td class="number">{comp_display}</td>
                        <td>{attempt.method}</td>
                        <td>{attempt.b1 or 'N/A'}</td>
                        <td>{attempt.curves_completed}</td>
                        <td>{factor_display}</td>
                        <td class="small-text">{attempt.client_id or 'Unknown'}</td>
                        <td>{attempt.execution_time_seconds:.1f}s</td>
                        <td class="small-text">{attempt.created_at.strftime('%Y-%m-%d %H:%M')}</td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>All Factors Found</h2>
            <table>
                <thead>
                    <tr>
                        <th>Factor</th>
                        <th>Composite</th>
                        <th>Discovery Method</th>
                        <th>Discovered</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for factor in factors:
        comp = db.query(Composite).filter(Composite.id == factor.composite_id).first()
        attempt = db.query(ECMAttempt).filter(ECMAttempt.id == factor.found_by_attempt_id).first()
        
        comp_display = comp.number[:40] + '...' if comp and len(comp.number) > 40 else (comp.number if comp else 'Unknown')
        method = attempt.method if attempt else 'Unknown'
        
        html_content += f"""
                    <tr>
                        <td class="factor number">{factor.factor}</td>
                        <td class="number">{comp_display}</td>
                        <td>{method}</td>
                        <td class="small-text">{factor.created_at.strftime('%Y-%m-%d %H:%M')}</td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
        </div>

        <div class="small-text">
            <p>Dashboard auto-refreshes every 30 seconds. API Documentation: <a href="/docs">/docs</a></p>
        </div>

        <script>
            // Search function for composites table
            function searchTable() {{
                const input = document.getElementById('searchInput');
                const filter = input.value.toLowerCase();
                const table = document.getElementById('compositesTable');
                const rows = table.getElementsByTagName('tr');

                // Loop through all table rows (skip header row)
                for (let i = 1; i < rows.length; i++) {{
                    const row = rows[i];
                    const cells = row.getElementsByTagName('td');
                    let found = false;

                    // Search in number column (first column)
                    if (cells.length > 0) {{
                        const numberText = cells[0].textContent || cells[0].innerText;
                        if (numberText.toLowerCase().indexOf(filter) > -1) {{
                            found = true;
                        }}
                    }}

                    // Show/hide row based on search
                    if (found || filter === '') {{
                        row.style.display = '';
                    }} else {{
                        row.style.display = 'none';
                    }}
                }}
            }}

            // Auto-refresh every 30 seconds
            setTimeout(function() {{
                window.location.reload();
            }}, 30000);
        </script>
    </body>
    </html>
    """

    return html_content


@router.get("/composites/{composite_id}/details", response_class=HTMLResponse)
async def get_composite_details_public(
    composite_id: int,
    db: Session = Depends(get_db)
):
    """
    Public web page showing detailed information about a specific composite.
    """
    from ...services.composite_manager import CompositeManager
    import html

    composite_manager = CompositeManager()
    details = composite_manager.get_composite_details(db, composite_id)

    if not details:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Composite Not Found</title>
        </head>
        <body>
            <h1>Composite Not Found</h1>
            <p><a href="/api/v1/dashboard/">Back to Dashboard</a></p>
        </body>
        </html>
        """

    composite = details['composite']
    progress = details['progress']
    recent_attempts = details['recent_attempts']
    active_work = details['active_work']

    # Helper function to escape HTML - prevent XSS
    def esc(text):
        return html.escape(str(text))

    # Format numbers for display
    number_display = composite['number'][:50] + '...' if len(composite['number']) > 50 else composite['number']

    # Format t-level progress
    current_t = composite['current_t_level'] or 0.0
    target_t = composite['target_t_level'] or 0.0
    progress_percent = (current_t / target_t * 100) if target_t > 0 else 0.0

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Composite Details - {composite['id']}</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; margin: 40px; line-height: 1.6; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
            .card {{ background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .progress-bar-container {{ background: #e9ecef; height: 30px; border-radius: 15px; overflow: hidden; }}
            .progress-fill {{ background: #28a745; height: 100%; transition: width 0.3s ease; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 0.9em; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
            th {{ background: #f8f9fa; font-weight: 600; }}
            .btn {{ padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }}
            .btn-primary {{ background: #007bff; color: white; }}
            .btn-secondary {{ background: #6c757d; color: white; }}
            .number-display {{ font-family: monospace; word-break: break-all; font-size: 0.9em; background: #f8f9fa; padding: 8px; border-radius: 4px; }}
            .status-badge {{ padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }}
            .status-complete {{ background: #d4edda; color: #155724; }}
            .status-active {{ background: #fff3cd; color: #856404; }}
            .status-pending {{ background: #f8d7da; color: #721c24; }}
            .back-btn {{ margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="back-btn">
                <a href="/api/v1/dashboard/" class="btn btn-secondary">← Back to Dashboard</a>
            </div>

            <div class="header">
                <h1>Composite Details</h1>
                <p style="margin: 10px 0 5px 0;"><strong>ID:</strong> {composite['id']} | <strong>{composite['digit_length']} digits</strong></p>
                <p style="margin: 5px 0; font-family: monospace; font-size: 0.85em; word-break: break-all; background: rgba(255,255,255,0.2); padding: 10px; border-radius: 5px;">{esc(composite['current_composite'])}</p>
            </div>

            <div class="card">
                <h2>Composite Information</h2>
                <table>
                    <tr><th>ID</th><td>{composite['id']}</td></tr>
                    <tr><th>Original Form</th><td class="number-display">{esc(composite['number'])}</td></tr>
                    <tr><th>Current Composite</th><td class="number-display">{esc(composite['current_composite'][:100])}{'...' if len(composite['current_composite']) > 100 else ''}</td></tr>
                    <tr><th>Digit Length</th><td>{composite['digit_length']}</td></tr>
                    <tr><th>Priority</th><td>{composite['priority']}</td></tr>
                    <tr><th>Created</th><td>{esc(composite['created_at'])}</td></tr>
                    <tr><th>Last Updated</th><td>{esc(composite['updated_at'])}</td></tr>
                    <tr><th>Status</th><td>
                        {'<span class="status-badge status-complete">Fully Factored</span>' if composite['is_fully_factored'] else '<span class="status-badge status-active">Active</span>'}
                        {' <span class="status-badge status-complete">Prime</span>' if composite['is_prime'] else ''}
                    </td></tr>
                </table>
            </div>

            <div class="card">
                <h2>T-Level Progress</h2>
                <table>
                    <tr><th>Current T-Level</th><td>t{current_t:.2f}</td></tr>
                    <tr><th>Target T-Level</th><td>t{target_t:.2f}</td></tr>
                    <tr><th>Progress</th><td>
                        <div class="progress-bar-container">
                            <div class="progress-fill" style="width: {min(progress_percent, 100):.1f}%">{progress_percent:.1f}%</div>
                        </div>
                    </td></tr>
                </table>
            </div>

            <div class="card">
                <h2>Work Summary</h2>
                <table>
                    <tr><th>Total Attempts</th><td>{progress['total_attempts']}</td></tr>
                    <tr><th>Total ECM Curves</th><td>{progress['total_ecm_curves']:,}</td></tr>
                    <tr><th>P-1 Attempts</th><td>{progress['pm1_attempts']}</td></tr>
                    <tr><th>Factors Found</th><td>{len(progress['factors_found'])}</td></tr>
                </table>
                {('<p style="margin-top: 15px;"><strong>Factors:</strong> ' + ', '.join([f'<code>{esc(f)}</code>' for f in progress['factors_found']]) + '</p>') if progress['factors_found'] else ''}
            </div>

            {'<div class="card"><h2>Active Work Assignments</h2><table><tr><th>Client ID</th><th>Method</th><th>B1</th><th>B2</th><th>Curves Requested</th><th>Status</th><th>Expires</th></tr>' + ''.join(f'<tr><td>{esc(work["client_id"])}</td><td>{esc(work["method"])}</td><td>{work["b1"]:,}</td><td>{work["b2"]:,}</td><td>{work["curves_requested"]}</td><td>{esc(work["status"])}</td><td>{esc(work["expires_at"])}</td></tr>' for work in active_work) + '</table></div>' if active_work else ''}

            <div class="card">
                <h2>Recent ECM Attempts</h2>
                {'<table><tr><th>Method</th><th>B1</th><th>B2</th><th>Curves Completed</th><th>Factor Found</th><th>Client</th><th>Submitted</th></tr>' + ''.join(f'<tr><td>{esc(attempt["method"])}</td><td>{attempt["b1"]:,}</td><td>{attempt["b2"]:,}</td><td>{attempt["curves_completed"]:,}</td><td class="number-display">{esc(attempt["factor_found"]) if attempt["factor_found"] else "None"}</td><td>{esc(attempt["client_id"])}</td><td>{esc(attempt["created_at"])}</td></tr>' for attempt in recent_attempts) + '</table>' if recent_attempts else '<p>No attempts yet.</p>'}
            </div>
        </div>
    </body>
    </html>
    """