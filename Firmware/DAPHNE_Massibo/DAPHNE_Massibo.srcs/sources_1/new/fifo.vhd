-- xilinx_fifo.vhd
-- common clock generic FIFO for Xilinx FPGAs
-- uses one 36-kbit or 18-kbit BlockRAM
-- jamieson olsen <jamieson@fnal.gov>

library ieee;
use ieee.std_logic_1164.all;

library unisim;
use unisim.vcomponents.all;
library unimacro;
use unimacro.vcomponents.all;

entity xilinx_fifo is
generic(DATA_WIDTH : integer := 64; 
        FIFO_SIZE  : string := "36Kb"; -- 18Kb or 36Kb BlockRAM
        RDCOUNT_SIZE : integer := 10);
port(
    clk:   in  std_logic;
    rst:   in  std_logic;
    din:   in  std_logic_vector(DATA_WIDTH-1 downto 0);
    wr_en: in  std_logic;
    rd_en: in  std_logic;
    dout:  out std_logic_vector(DATA_WIDTH-1 downto 0);
    full:  out std_logic;
    empty: out std_logic
);
end xilinx_fifo;

architecture fifo_arch of xilinx_fifo is

signal RDCOUNT, WRCOUNT: std_logic_vector(RDCOUNT_SIZE-1 downto 0);

begin

-- FIFO_SYNC_MACRO: Synchronous First-In, First-Out (FIFO) RAM Buffer
-- Note - This Unimacro model assumes the port directions to be "downto".
-- Simulation of this model with "to" in the port directions could lead to erroneous results.
-----------------------------------------------------------------
-- DATA_WIDTH | FIFO_SIZE | FIFO Depth | RDCOUNT/WRCOUNT Width --
-- ===========|===========|============|=======================--
--   37-72    | "36Kb"    | 512        |  9-bit --
--   19-36    | "36Kb"    | 1024       | 10-bit --
--   19-36    | "18Kb"    | 512        |  9-bit --
--   10-18    | "36Kb"    | 2048       | 11-bit --
--   10-18    | "18Kb"    | 1024       | 10-bit --
--    5-9     | "36Kb"    | 4096       | 12-bit --
--    5-9     | "18Kb"    | 2048       | 11-bit --
--    1-4     | "36Kb"    | 8192       | 13-bit --
--    1-4     | "18Kb"    | 4096       | 12-bit --
-----------------------------------------------------------------

-- NOTE: in this macro the RDCOUNT and WRCOUNT output ports must
-- be connected to *something* or else this macro will generate errors!

FIFO_SYNC_MACRO_inst : FIFO_SYNC_MACRO
generic map(
    DEVICE => "7SERIES",            -- Target Device: "VIRTEX5, "VIRTEX6", "7SERIES"
    ALMOST_FULL_OFFSET  => X"0080", -- Sets almost full threshold
    ALMOST_EMPTY_OFFSET => X"0080", -- Sets the almost empty threshold
    DATA_WIDTH => DATA_WIDTH,       -- Valid values are 1-72 (37-72 only valid when FIFO_SIZE="36Kb")
    FIFO_SIZE => FIFO_SIZE)            -- Target BRAM, "18Kb" or "36Kb"
port map(
    ALMOSTEMPTY => open,     -- 1-bit output almost empty
    ALMOSTFULL  => open,     -- 1-bit output almost full
    DO          => dout,     -- Output data, width defined by DATA_WIDTH parameter
    EMPTY       => empty,    -- 1-bit output empty
    FULL        => full,     -- 1-bit output full
    RDCOUNT     => RDCOUNT,  -- Output read count, width determined by FIFO depth
    RDERR       => open,     -- 1-bit output read error
    WRCOUNT     => WRCOUNT,  -- Output write count, width determined by FIFO depth
    WRERR       => open,     -- 1-bit output write error
    CLK         => clk,      -- common clock
    DI          => din,      -- data in
    RDEN        => rd_en,    -- read enable
    RST         => rst,      -- sync reset
    WREN        => wr_en     -- write enable
);

end fifo_arch;
