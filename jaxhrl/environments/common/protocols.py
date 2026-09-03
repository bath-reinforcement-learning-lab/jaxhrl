from abc import abstractmethod
from collections.abc import Iterable, Sequence
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np

ObsType = TypeVar("ObsType", contravariant=True)
ActType = TypeVar("ActType", covariant=True)


@runtime_checkable
class ImplementsGetActionMask(Protocol[ActType, ObsType]):
    """Environment protocol for checking available actions.
    - get_available_actions: Returns a list of available actions in the current state of the environment.
    """

    @abstractmethod
    def get_action_mask(self, obs: ObsType | None = None) -> np.ndarray: ...


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType", contravariant=True)


@runtime_checkable
class ImplementsGetSuccessors(Protocol[ActType, ObsType]):
    """Environment protocol for if the transition dynamics are obtainable. Specifically, the `get_successors` method must be implemented.
    - get_successors: (State, Action) -> List[(Next State, Reward), Probability]
    """

    @abstractmethod
    def get_successors(
        self, state: ObsType | None = None, action: ActType | None = None
    ) -> Iterable[tuple[tuple[ObsType, float], float]]: ...


ActType = TypeVar("ActType", contravariant=True)


@runtime_checkable
class ImplementsGetAllStates(Protocol[ObsType]):
    """Environment protocol for if the entire state space can be generated.
    - get_all_states: Returns a set of every possible state an agent could be in.
    """

    @abstractmethod
    def get_all_states(self) -> set[ObsType]: ...


ObsType = TypeVar("ObsType", contravariant=True)


@runtime_checkable
class ImplementsIsStateTerminal(Protocol[ObsType]):
    """Environment protocol for if states can be tested to check if they are terminal.
    - is_state_terminal: Checks if a given state would terminate the environment.
    """

    @abstractmethod
    def is_state_terminal(self, state: ObsType | None = None) -> bool: ...


ObsType = TypeVar("ObsType", covariant=True)


@runtime_checkable
class ImplementsGetInitialStates(Protocol[ObsType]):
    """Environment protocol to return all possible initial states.
    - get_initial_states: Returns a sequence of all initial states.
    """

    @abstractmethod
    def get_initial_states(self) -> Sequence[ObsType]: ...
