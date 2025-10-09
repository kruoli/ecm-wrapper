"""
Group Order Calculator Service

Calculates elliptic curve group orders for ECM factors using PARI/GP.
Based on the FindGroupOrder function for different ECM parametrizations.
"""

import subprocess
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class GroupOrderCalculator:
    """Calculate elliptic curve group orders for ECM factors."""

    def __init__(self, gp_binary: str = "gp", script_path: Optional[str] = None):
        """
        Initialize group order calculator.

        Args:
            gp_binary: Path to PARI/GP binary (default: "gp" in PATH)
            script_path: Path to group.gp script (default: /app/bin/group.gp or ./bin/group.gp)
        """
        self.gp_binary = gp_binary

        # Find the group.gp script
        if script_path:
            self.script_path = script_path
        else:
            # Try Docker location first, then local dev location
            possible_paths = [
                "/app/bin/group.gp",
                Path(__file__).parent.parent.parent / "bin" / "group.gp",
            ]
            self.script_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    self.script_path = str(path)
                    break

            if not self.script_path:
                logger.warning("group.gp script not found, will use inline script")
                self.script_path = None

    def calculate_group_order(
        self, factor: str, sigma: str, parametrization: int = 3
    ) -> Optional[Tuple[str, str]]:
        """
        Calculate elliptic curve group order for a factor found by ECM.

        Args:
            factor: Prime factor (p)
            sigma: Sigma value used to find the factor
            parametrization: ECM parametrization (0, 1, 2, or 3)

        Returns:
            Tuple of (group_order, factorization) or None if calculation fails
        """
        # Parse sigma if it includes parametrization prefix (e.g., "3:12345")
        sigma_value = sigma
        if isinstance(sigma, str) and ':' in sigma:
            parts = sigma.split(':', 1)
            # Use parametrization from sigma if not explicitly provided
            if parametrization == 3:  # Default value
                try:
                    parametrization = int(parts[0])
                except ValueError:
                    pass
            sigma_value = parts[1]

        # Validate parametrization
        if parametrization not in [0, 1, 2, 3]:
            logger.warning(
                f"Invalid parametrization {parametrization}, defaulting to 3"
            )
            parametrization = 3

        # Build PARI/GP script to load function and call it
        if self.script_path:
            # Use external script file
            script = f'read("{self.script_path}");FindGroupOrder({factor},{sigma_value},{parametrization})\nquit\n'
        else:
            # Fallback to inline condensed version
            inline_script = 'FindGroupOrder(p,s,param=0)={{A=0;b=0;if(param==0,v=Mod(4*s,p);u=Mod(s^2-5,p);x=u^3;A=(3*u+v)*(v-u)^3/(4*x*v)-2;x=x/v^3;b=x*(x*(x+A)+1),param==1,A=Mod(4*s^2,p)/2^64-2;b=4*A+10,param==2,E=ellinit([0,Mod(36,p)]);[x,y]=ellmul(E,[-3,3],s);x3=(3*x+y+6)/(2*(y-3));A=-(3*x3^4+6*x3^2-1)/(4*x3^3);b=1/(4*A+10),param==3,A=Mod(4*s,p)/2^32-2;b=4*A+10);if(param>=0&&param<=3,E=ellinit([0,b*A,0,b^2,0]);ellcard(E),0)}};'
            script = f'{inline_script}FindGroupOrder({factor},{sigma_value},{parametrization})\nquit\n'

        try:
            # Execute PARI/GP with explicit encoding
            result = subprocess.run(
                [self.gp_binary, "-q", "-f"],  # -q: quiet, -f: fast
                input=script.encode('utf-8'),
                capture_output=True,
                timeout=30,  # 30 second timeout
            )

            if result.returncode != 0:
                stderr = result.stderr.decode('utf-8') if isinstance(result.stderr, bytes) else result.stderr
                logger.error(f"PARI/GP error: {stderr}")
                return None

            # Parse output - last line should be the group order
            stdout = result.stdout.decode('utf-8') if isinstance(result.stdout, bytes) else result.stdout
            output = stdout.strip()
            if not output:
                logger.error("PARI/GP returned no output")
                return None

            # Extract the group order (last line of output)
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            if not lines:
                logger.error("No valid output from PARI/GP")
                return None

            group_order = lines[-1]

            # Try to factor the group order for interesting structure
            factorization = self._factor_group_order(group_order)

            logger.info(
                f"Calculated group order for factor {factor[:20]}... "
                f"with sigma {sigma_value}: {group_order}"
            )

            return (group_order, factorization)

        except subprocess.TimeoutExpired:
            logger.error(
                f"PARI/GP timeout calculating group order for factor {factor[:20]}..."
            )
            return None
        except Exception as e:
            logger.error(
                f"Error calculating group order for factor {factor[:20]}...: {e}"
            )
            return None

    def _factor_group_order(self, group_order: str) -> Optional[str]:
        """
        Factor the group order using PARI/GP.

        Args:
            group_order: The group order to factor

        Returns:
            Factorization string in format "2^5 * 3^2 * 5^2 * ..." or None if factorization fails
        """
        script = f"factor({group_order})\nquit\n"

        try:
            result = subprocess.run(
                [self.gp_binary, "-q", "-f"],
                input=script.encode('utf-8'),
                capture_output=True,
                timeout=10,
            )

            if result.returncode != 0:
                return None

            stdout = result.stdout.decode('utf-8') if isinstance(result.stdout, bytes) else result.stdout
            output = stdout.strip()
            if not output:
                return None

            # Parse PARI/GP matrix format: [2 5]\n[3 2]\n[5 2] means 2^5 * 3^2 * 5^2
            factors = []
            for line in output.split('\n'):
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    # Remove brackets and split
                    parts = line[1:-1].split()
                    if len(parts) == 2:
                        base = parts[0]
                        exp = parts[1]
                        if exp == '1':
                            factors.append(base)
                        else:
                            factors.append(f"{base}^{exp}")

            if factors:
                return " * ".join(factors)

            return None

        except Exception:
            return None
