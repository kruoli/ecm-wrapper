"""
Typed Configuration Classes

Provides type-safe access to configuration values, replacing
dictionary-based access with proper dataclasses. This enables:
- IDE autocompletion and type checking
- Validation at config load time
- Clear documentation of available settings
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class APIEndpoint:
    """Configuration for a single API endpoint."""
    url: str
    name: str = "default"


@dataclass
class APIConfig:
    """API connection configuration."""
    endpoint: str = "http://localhost:8000/api/v1"
    endpoints: List[APIEndpoint] = field(default_factory=list)
    retry_attempts: int = 3
    timeout: int = 30

    def get_endpoints(self) -> List[APIEndpoint]:
        """
        Get list of endpoints to submit to.

        Returns endpoints list if configured, otherwise wraps
        the single endpoint in a list.
        """
        if self.endpoints:
            return self.endpoints
        return [APIEndpoint(url=self.endpoint, name="default")]


@dataclass
class ClientConfig:
    """Client identification configuration."""
    username: str = "default_user"
    cpu_name: str = "default_machine"


@dataclass
class ExecutionConfig:
    """Execution environment configuration."""
    output_dir: str = "data/outputs"
    residue_dir: str = "data/residues"
    failed_uploads_dir: str = "data/failed_uploads"
    preserve_failed_uploads: bool = True
    save_raw_output: bool = True

    def ensure_dirs_exist(self) -> None:
        """Create output directories if they don't exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.residue_dir).mkdir(parents=True, exist_ok=True)
        Path(self.failed_uploads_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    file: str = "data/logs/ecm_client.log"
    level: str = "INFO"
    log_factors_found: bool = True

    def ensure_log_dir_exists(self) -> None:
        """Create log directory if it doesn't exist."""
        Path(self.file).parent.mkdir(parents=True, exist_ok=True)


@dataclass
class GMPECMConfig:
    """GMP-ECM program configuration."""
    path: str = "ecm"
    default_b1: int = 110000000
    default_b2: Optional[int] = None
    default_curves: int = 1
    early_termination: bool = True
    gpu_enabled: bool = False
    gpu_device: int = 0
    gpu_curves: Optional[int] = None
    workers: int = 8  # Parallel workers (multiprocess ECM, stage2 threads)
    pm1_b1: int = 2900000000
    pm1_b2: int = 1000000000000000
    pp1_b1: int = 110000000
    pp1_b2: int = 500000000000


@dataclass
class YAFUConfig:
    """YAFU program configuration."""
    path: str = "yafu"
    threads: int = 8


@dataclass
class CADOConfig:
    """CADO-NFS program configuration."""
    path: str = "~/cado-nfs/cado-nfs.py"
    threads: int = 4
    working_dir: str = "~/cado-nfs"


@dataclass
class TLevelConfig:
    """T-level calculator configuration."""
    path: str = "bin/t-level"


@dataclass
class ProgramsConfig:
    """All program configurations."""
    gmp_ecm: GMPECMConfig = field(default_factory=GMPECMConfig)
    yafu: YAFUConfig = field(default_factory=YAFUConfig)
    cado_nfs: CADOConfig = field(default_factory=CADOConfig)
    t_level: TLevelConfig = field(default_factory=TLevelConfig)


@dataclass
class AppConfig:
    """
    Root configuration object containing all settings.

    This is the main configuration class that aggregates all
    sub-configurations. Use TypedConfigLoader to create instances
    from YAML files.

    Usage:
        config = TypedConfigLoader().load("client.yaml")
        print(config.programs.gmp_ecm.path)
        print(config.api.endpoint)
    """
    api: APIConfig = field(default_factory=APIConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    programs: ProgramsConfig = field(default_factory=ProgramsConfig)

    def ensure_dirs_exist(self) -> None:
        """Create all required directories."""
        self.execution.ensure_dirs_exist()
        self.logging.ensure_log_dir_exists()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config back to dictionary format.

        This is useful for backward compatibility with code
        that still expects dictionary access.
        """
        return {
            'api': {
                'endpoint': self.api.endpoint,
                'endpoints': [{'url': e.url, 'name': e.name} for e in self.api.endpoints],
                'retry_attempts': self.api.retry_attempts,
                'timeout': self.api.timeout,
            },
            'client': {
                'username': self.client.username,
                'cpu_name': self.client.cpu_name,
            },
            'execution': {
                'output_dir': self.execution.output_dir,
                'residue_dir': self.execution.residue_dir,
                'failed_uploads_dir': self.execution.failed_uploads_dir,
                'preserve_failed_uploads': self.execution.preserve_failed_uploads,
                'save_raw_output': self.execution.save_raw_output,
            },
            'logging': {
                'file': self.logging.file,
                'level': self.logging.level,
                'log_factors_found': self.logging.log_factors_found,
            },
            'programs': {
                'gmp_ecm': {
                    'path': self.programs.gmp_ecm.path,
                    'default_b1': self.programs.gmp_ecm.default_b1,
                    'default_b2': self.programs.gmp_ecm.default_b2,
                    'default_curves': self.programs.gmp_ecm.default_curves,
                    'early_termination': self.programs.gmp_ecm.early_termination,
                    'gpu_enabled': self.programs.gmp_ecm.gpu_enabled,
                    'gpu_device': self.programs.gmp_ecm.gpu_device,
                    'gpu_curves': self.programs.gmp_ecm.gpu_curves,
                    'workers': self.programs.gmp_ecm.workers,
                    'pm1_b1': self.programs.gmp_ecm.pm1_b1,
                    'pm1_b2': self.programs.gmp_ecm.pm1_b2,
                    'pp1_b1': self.programs.gmp_ecm.pp1_b1,
                    'pp1_b2': self.programs.gmp_ecm.pp1_b2,
                },
                'yafu': {
                    'path': self.programs.yafu.path,
                    'threads': self.programs.yafu.threads,
                },
                'cado_nfs': {
                    'path': self.programs.cado_nfs.path,
                    'threads': self.programs.cado_nfs.threads,
                    'working_dir': self.programs.cado_nfs.working_dir,
                },
                't_level': {
                    'path': self.programs.t_level.path,
                },
            },
        }


class TypedConfigLoader:
    """
    Load configuration from YAML into typed dataclasses.

    This class wraps ConfigManager to provide type-safe access
    to configuration values.

    Usage:
        loader = TypedConfigLoader()
        config = loader.load("client.yaml")
        print(config.programs.gmp_ecm.path)
    """

    def load(self, config_path: str) -> AppConfig:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Typed AppConfig instance
        """
        from .config_manager import ConfigManager

        # Load raw config using existing ConfigManager
        manager = ConfigManager()
        raw_config = manager.load_config(config_path)

        return self._parse_config(raw_config)

    def _parse_config(self, raw: Dict[str, Any]) -> AppConfig:
        """Parse raw dictionary into typed config."""
        return AppConfig(
            api=self._parse_api(raw.get('api', {})),
            client=self._parse_client(raw.get('client', {})),
            execution=self._parse_execution(raw.get('execution', {})),
            logging=self._parse_logging(raw.get('logging', {})),
            programs=self._parse_programs(raw.get('programs', {})),
        )

    def _parse_api(self, raw: Dict[str, Any]) -> APIConfig:
        """Parse API configuration."""
        endpoints = []
        if 'endpoints' in raw:
            for ep in raw['endpoints']:
                endpoints.append(APIEndpoint(
                    url=ep.get('url', ''),
                    name=ep.get('name', 'default'),
                ))

        return APIConfig(
            endpoint=raw.get('endpoint', 'http://localhost:8000/api/v1'),
            endpoints=endpoints,
            retry_attempts=raw.get('retry_attempts', 3),
            timeout=raw.get('timeout', 30),
        )

    def _parse_client(self, raw: Dict[str, Any]) -> ClientConfig:
        """Parse client configuration."""
        return ClientConfig(
            username=raw.get('username', 'default_user'),
            cpu_name=raw.get('cpu_name', 'default_machine'),
        )

    def _parse_execution(self, raw: Dict[str, Any]) -> ExecutionConfig:
        """Parse execution configuration."""
        return ExecutionConfig(
            output_dir=raw.get('output_dir', 'data/outputs'),
            residue_dir=raw.get('residue_dir', 'data/residues'),
            failed_uploads_dir=raw.get('failed_uploads_dir', 'data/failed_uploads'),
            preserve_failed_uploads=raw.get('preserve_failed_uploads', True),
            save_raw_output=raw.get('save_raw_output', True),
        )

    def _parse_logging(self, raw: Dict[str, Any]) -> LoggingConfig:
        """Parse logging configuration."""
        return LoggingConfig(
            file=raw.get('file', 'data/logs/ecm_client.log'),
            level=raw.get('level', 'INFO'),
            log_factors_found=raw.get('log_factors_found', True),
        )

    def _parse_programs(self, raw: Dict[str, Any]) -> ProgramsConfig:
        """Parse programs configuration."""
        return ProgramsConfig(
            gmp_ecm=self._parse_gmp_ecm(raw.get('gmp_ecm', {})),
            yafu=self._parse_yafu(raw.get('yafu', {})),
            cado_nfs=self._parse_cado(raw.get('cado_nfs', {})),
            t_level=self._parse_tlevel(raw.get('t_level', {})),
        )

    def _parse_gmp_ecm(self, raw: Dict[str, Any]) -> GMPECMConfig:
        """Parse GMP-ECM configuration."""
        return GMPECMConfig(
            path=raw.get('path', 'ecm'),
            default_b1=raw.get('default_b1', 110000000),
            default_b2=raw.get('default_b2'),
            default_curves=raw.get('default_curves', 1),
            early_termination=raw.get('early_termination', True),
            gpu_enabled=raw.get('gpu_enabled', False),
            gpu_device=raw.get('gpu_device', 0),
            gpu_curves=raw.get('gpu_curves'),
            workers=raw.get('workers', raw.get('stage2_workers', 8)),  # Backward compat
            pm1_b1=raw.get('pm1_b1', 2900000000),
            pm1_b2=raw.get('pm1_b2', 1000000000000000),
            pp1_b1=raw.get('pp1_b1', 110000000),
            pp1_b2=raw.get('pp1_b2', 500000000000),
        )

    def _parse_yafu(self, raw: Dict[str, Any]) -> YAFUConfig:
        """Parse YAFU configuration."""
        return YAFUConfig(
            path=raw.get('path', 'yafu'),
            threads=raw.get('threads', 8),
        )

    def _parse_cado(self, raw: Dict[str, Any]) -> CADOConfig:
        """Parse CADO-NFS configuration."""
        return CADOConfig(
            path=raw.get('path', '~/cado-nfs/cado-nfs.py'),
            threads=raw.get('threads', 4),
            working_dir=raw.get('working_dir', '~/cado-nfs'),
        )

    def _parse_tlevel(self, raw: Dict[str, Any]) -> TLevelConfig:
        """Parse t-level configuration."""
        return TLevelConfig(
            path=raw.get('path', 'bin/t-level'),
        )
