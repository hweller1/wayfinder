// Fixture: top-level declarations of a C# file.
using System;
using System.Collections.Generic;

namespace Fixture;

public class Inventory
{
    private readonly Dictionary<string, int> stock = new();

    public void Add(string sku, int quantity)
    {
        stock[sku] = stock.GetValueOrDefault(sku) + quantity;
    }

    public int Remove(string sku, int quantity)
    {
        var have = stock.GetValueOrDefault(sku);
        var taken = Math.Min(have, quantity);
        stock[sku] = have - taken;
        return taken;
    }
}

public interface IAuditor
{
    void Record(string sku, int delta, DateTime when);
}
