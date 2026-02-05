# Proactive Query Capabilities

This document outlines all the proactive query types that Jarvis now supports, organized by category.

## Investment Analysis Queries

### Equity Allocation & Portfolio Analysis
- **"Which of my clients are underweight in equities relative to their risk profile and time horizon?"**
  - Identifies clients whose current equity allocation is below their target range based on risk profile
  - Shows gap between current and target allocation
  - Includes time horizon and portfolio value for context

### Tax Allowances
- **"Show me everyone with ISA allowance still available this tax year"**
  - Lists clients with remaining ISA allowance (2024/25: £20,000 limit)
  - Shows used vs available amounts
  
- **"Show me everyone with Annual allowance still available this tax year"**
  - Lists clients with remaining pension annual allowance
  - Accounts for tapered allowances where applicable

### Cash Management
- **"Which clients have cash excess above 6 months expenditure that we should discuss investing?"**
  - Identifies clients holding excessive cash
  - Shows cash holdings in months of expenditure
  - Highlights investment opportunities

### Retirement Planning
- **"Flag any clients where their current trajectory won't meet their stated retirement income goals"**
  - Analyzes retirement income adequacy
  - Compares current portfolio trajectory to stated goals
  - Identifies clients needing plan adjustments

- **"Which retired clients are taking more than 4% withdrawal rates?"**
  - Flags sustainability concerns
  - Shows withdrawal rate and portfolio value

### Protection Analysis
- **"Which clients have protection gaps based on their family circumstances?"**
  - Identifies clients with dependents but insufficient protection
  - Considers life insurance, critical illness, and income protection

### Scenario Analysis
- **"Show me which clients would be impacted if interest rates drop to 3%"**
  - Uses search to find clients with interest rate sensitivity
  
- **"Model what happens to the Gurung's family's plan if one of them needs long-term care"**
  - Uses client-specific search and analysis
  
- **"Which clients are most exposed if we see a 20% market correction?"**
  - Identifies high equity allocation clients
  
- **"If Roshan retires next year instead of in three years, what does their cashflow look like?"**
  - Uses client-specific analysis

## Proactive Client Identification

### Review Management
- **"Which clients haven't had a review in over 12 months?"**
  - Lists clients overdue for annual review
  - Shows days overdue and last review date

### Business Owners
- **"Show me all business owners who might benefit from the new R&D tax credit changes"**
  - Identifies business owner clients
  - Shows business type and age

### Education Planning
- **"Who has children approaching university age but no education planning in place?"**
  - Identifies clients with children aged 16-19
  - Flags missing education planning

### Client Matching
- **"Find clients with similar profiles to the Smiths who successfully navigated early retirement"**
  - Uses semantic search to find similar client profiles
  - Helps identify case studies and success patterns

### Estate Planning
- **"Which high-net-worth clients don't have estate planning in place?"**
  - Identifies HNW clients (typically >£1M)
  - Shows will and trust status

### Service Opportunities
- **"Show me pension clients who might benefit from our cashflow modelling service"**
  - Identifies pension clients not yet retired
  - Highlights service opportunities

- **"Who has investment portfolios but no protection cover?"**
  - Cross-selling opportunities
  - Shows portfolio value and dependents

- **"Which business owner clients haven't discussed exit planning?"**
  - Identifies business owners without succession planning discussions

### Milestones
- **"Which clients have birthdays this month?"**
  - Monthly birthday tracking
  - Shows age and turning age

## Compliance & Documentation

### Recommendation Tracking
- **"Pull every recommendation I made to David Chen and the rationale I gave"**
  - Complete recommendation history for a client
  - Includes rationale and status

### Conversation Search
- **"What was my exact wording when discussing risk with the Williams family?"**
  - Searches meeting transcripts for specific wording
  - Shows exact conversation excerpts

- **"Show me all clients where I recommended Platform X and why"**
  - Tracks platform recommendations
  - Shows rationale for each recommendation

### Topic Analysis
- **"Which client conversations mentioned concerns about market volatility?"**
  - Searches conversations by topic
  - Lists all relevant meetings

- **"Generate a summary of all discussions about sustainable investing preferences"**
  - Summarizes discussions by topic
  - Groups by client and date

### Document Management
- **"What documents am I still waiting for from clients?"**
  - Lists all outstanding document requests
  - Shows client, document type, and request date

- **"What did I promise to send the Jackson family and when?"**
  - Tracks promises and commitments
  - Shows status (pending/overdue)

## Business Analytics

### Client Concerns
- **"What concerns clients raised in meetings this month?"**
  - Monthly concerns analysis
  - Groups by concern type

### Service Usage
- **"Which services do my highest-value clients use most?"**
  - Analyzes product usage by high-value clients
  - Shows percentage breakdown

### Conversion Analysis
- **"Show me conversion rates from initial meeting to becoming a client by referral source"**
  - Conversion metrics by source
  - Helps optimize marketing efforts

### Book Demographics
- **"What percentage of my book is approaching retirement in the next 5 years?"**
  - Book composition analysis
  - Shows age distribution and client segments

### Efficiency Analysis
- **"Which clients generate the most revenue but take the least time to service?"**
  - Revenue vs time efficiency
  - Identifies most efficient client relationships

### Client Patterns
- **"What do my most satisfied long-term clients have in common?"**
  - Identifies success patterns
  - Shows common characteristics

- **"Which types of recommendations get the most pushback and why?"**
  - Recommendation acceptance analysis
  - Helps refine approach

- **"Show me clients whose circumstances are similar to cases where we added significant value"**
  - Case study matching
  - Identifies similar opportunities

- **"What life events trigger clients to actually implement recommendations?"**
  - Implementation trigger analysis
  - Shows which events drive action

## Follow-up & Actions

### Email Drafting
- **"Draft the follow-up email to yesterday's meeting with the key actions we agreed"**
  - Generates follow-up email template
  - Includes action items and next steps

### Waiting Items
- **"Which clients am I waiting on for information or decisions?"**
  - Lists all pending items
  - Shows documents and actions waiting

### Action Management
- **"Show me all open action items across my client base"**
  - Complete action item list
  - Sorted by due date

- **"What follow-ups did I commit to that are now overdue?"**
  - Overdue follow-up tracking
  - Shows days overdue

## Usage Tips

1. **Be Specific**: The more specific your query, the better the results. Include client names when asking about specific clients.

2. **Use Natural Language**: Jarvis understands natural language queries. You don't need to use exact phrases.

3. **Combine Queries**: You can ask follow-up questions to drill deeper into results.

4. **Proactive Insights**: Jarvis will proactively suggest related queries and opportunities.

5. **Data-Driven**: All responses are based on actual client data, ensuring accuracy and relevance.

## Technical Notes

- All queries use semantic search across client profiles and meeting transcripts
- Investment analysis uses calculated fields from client data
- Compliance tracking monitors deadlines and commitments automatically
- Business analytics aggregate data across the entire client base
- Follow-up tools integrate with meeting notes and action items

## Future Enhancements

- Real-time data integration with CRM systems
- Predictive analytics for client behavior
- Automated scenario modeling
- Advanced cashflow projections
- Integration with portfolio platforms for real-time data

